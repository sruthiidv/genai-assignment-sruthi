"""Build plots and readable summaries directly from saved experiment outputs."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def markdown_table(frame):
    rows=[list(frame.columns)]+[[f'{v:.4f}' if isinstance(v,float) else str(v) for v in row] for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(r)+' |' for r in rows[1:]])

def build_report(out):
    out=Path(out); metrics=pd.read_csv(out/'metrics.csv')
    samples=pd.read_json(out/'samples.json')
    histories=json.loads((out/'histories.json').read_text())
    plt.rcParams.update({'figure.dpi':140,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,(kind,rows) in zip(axs.flat,histories.items()):
        ax.plot([r['epoch'] for r in rows],[r['train_loss'] for r in rows],label='Train')
        ax.plot([r['epoch'] for r in rows],[r['validation_loss'] for r in rows],label='Validation')
        ax.set(title=kind,xlabel='Epoch',ylabel='Cross-entropy (nats)'); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Training and validation loss'); fig.savefig(out/'learning_curves.png'); plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,col,title in zip(axs,['perplexity','accuracy','training_seconds'],['Test perplexity (lower is better)','Test accuracy (fraction)','Training time (seconds)']):
        ax.bar(metrics.model,metrics[col],color=['#4477aa','#ee6677','#228833','#aa3377'])
        ax.set_title(title); ax.tick_params(axis='x',rotation=25)
    fig.savefig(out/'model_comparison.png'); plt.close(fig)
    group=samples.groupby(['model','temperature'])[['distinct_1','distinct_2','repeated_trigram_rate']].mean().reset_index()
    group.to_csv(out/'temperature_summary.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,col in zip(axs,['distinct_1','distinct_2','repeated_trigram_rate']):
        for model,rows in group.groupby('model'):
            ax.plot(rows.temperature,rows[col],marker='o',label=model)
        ax.set(xlabel='Temperature',ylabel=col.replace('_',' '),xticks=[.5,1,1.5],ylim=(-.03,1.03)); ax.grid(alpha=.2)
    axs[0].legend(fontsize=8); fig.savefig(out/'temperature_effects.png'); plt.close(fig)
    best=metrics.sort_values('perplexity').iloc[0]
    overview=f'''# Recorded experiment results

{markdown_table(metrics)}

The lowest test perplexity in this run belongs to **{best['model']}** ({best['perplexity']:.2f}), with test accuracy {best['accuracy']:.2%}. This is the quantitative winner for this dataset and split, not a claim of universal superiority.

## Temperature statistics

Each row averages nine samples: three prompts × three sampling seeds. Diversity/repetition are measured on the 100 generated words only, excluding the prompt.

{markdown_table(group)}

Distinct-1 and distinct-2 are unique unigram/bigram counts divided by their total counts. Repeated-trigram rate is one minus unique trigrams divided by total trigrams. These describe lexical behaviour, not semantic coherence. High diversity can also reflect incoherent word choices.

## Interpretation limits

Only one model-initialization seed was trained. A small single-book corpus, chronological topic shifts, a fixed 20-word context, removal of punctuation, and vocabulary mapping to `<unk>` limit interpretation. Perplexity includes unknown targets, so it is not directly comparable to a different vocabulary. Parameter counts differ even with equal hidden sizes. Early stopping means different actual training durations under the same budget. Training time includes validation and checkpoint writes but excludes generation and test evaluation.

See qualitative_review.md for the fixed representative samples and semantic assessment. All 108 samples are preserved in samples.json and samples.csv.
'''
    (out/'summary.md').write_text(overview,encoding='utf-8')
    selected=samples[(samples.seed_index==0)&(samples.sampling_seed==101)]
    text=['# Fixed representative samples','Selected in advance: prompt 0 and sampling seed 101 for every model and temperature.','Prompt: '+selected.iloc[0]['prompt']]
    for row in selected.itertuples():
        text += [f'## {row.model} — temperature {row.temperature}',row.generated]
    (out/'representative_samples.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')

if __name__=='__main__':
    build_report(Path(__file__).resolve().parent/'results/full')
