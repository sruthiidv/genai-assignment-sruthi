"""Create the teaching notebook; execution is a separate explicit step."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parent
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
nb=nbf.v4.new_notebook(cells=[
md('''# Alice in Wonderland: Four Autoregressive Text Generators
**Foundations of Generative AI (ICT4403), Assignment 2**

We compare Simple RNN, Bidirectional RNN, LSTM and GRU for word-level next-word prediction. Training uses a common dataset, optimizer and stopping policy. The best checkpoint is chosen by validation loss; test data is used only for final evaluation. All results below come from saved runs, not invented numbers.

Run this notebook from the project directory. Install `requirements.txt` first. Set `RETRAIN = True` to reproduce training; the default loads the supplied results so opening the notebook does not unnecessarily repeat training.'''),
code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Markdown, Image
from experiment import ROOT, CONFIG, prepare, run, TextModel, KINDS
RETRAIN = False
OUT = ROOT / 'results' / 'full'
display(CONFIG)'''),
md('''## 1. Dataset and preprocessing
Source: [Project Gutenberg ebook 11](https://www.gutenberg.org/ebooks/11), Lewis Carroll's *Alice's Adventures in Wonderland*. The retained raw download includes its source licence; a SHA-256 checksum identifies the exact bytes.

Remove Gutenberg boilerplate, the contents and chapter headings. Normalize Unicode punctuation and lowercase. Extract words with internal apostrophes; punctuation is not predicted. Split the ordered token stream 80/10/10 **before** windowing. Build the vocabulary only from training words appearing at least twice; other words become `<unk>`. Each example has 20 preceding words and one next-word label. Windows cannot cross split boundaries.'''),
code('''datasets, vocabulary, prompts = prepare()
display(json.loads((ROOT/'data/provenance.json').read_text()))
stats = json.loads((ROOT/'data/statistics.json').read_text())
print('Total words:', stats['total_words'], '| Vocabulary:', stats['vocabulary_size'])
display(pd.DataFrame(stats['splits']).T)
x, y = datasets['train'][0]
print('Input:', ' '.join(vocabulary[i] for i in x.tolist()))
print('Target:', vocabulary[y.item()])
display(pd.DataFrame({'prompt_id': range(3), 'seed_text': [' '.join(p) for p in prompts]}))'''),
md('''## 2. Models and fair training
All models use a 64-dimensional word embedding, one recurrent layer with hidden size 128 per direction, and a linear vocabulary classifier. Simple RNN uses a tanh recurrence; LSTM and GRU use gates to control memory. The Bidirectional RNN concatenates the final forward and backward hidden states of the **known preceding context only**. It never sees the target word.

Adam uses learning rate 0.001, batch size 64, gradient clipping at norm 1, and a maximum of 25 epochs. Early stopping waits four epochs without validation-loss improvement. All runs start from seed 42 and use the same batch-order seed. Equal hidden sizes do not imply equal parameter counts. Different stopping epochs are a consequence of the shared selection rule, not separate training budgets.'''),
code('''display(pd.DataFrame([{'model': k, 'parameters': sum(p.numel() for p in TextModel(k, len(vocabulary)).parameters())} for k in KINDS]))
if RETRAIN or not (OUT/'metrics.csv').exists():
    run(smoke=False)
else:
    print('Loading recorded full-run results. Set RETRAIN=True to train again.')
display(json.loads((OUT/'environment.json').read_text()))'''),
md('''## 3. Held-out evaluation
Cross-entropy is averaged over individual test examples. Perplexity is `exp(loss)` using natural logarithms. Accuracy is the fraction whose highest-probability word matches the target. Unknown targets are included, which must be remembered when comparing against other vocabularies. Time includes training, validation and checkpoint writes, with GPU synchronization if applicable.'''),
code('''metrics = pd.read_csv(OUT/'metrics.csv')
display(metrics)
display(Image(filename=str(OUT/'learning_curves.png')))
display(Image(filename=str(OUT/'model_comparison.png')))'''),
md('''## 4. Temperature experiment
For each model, generate 100 words for each of three fixed 20-word training prompts, temperatures 0.5, 1.0 and 1.5, and sampling seeds 101, 202 and 303: **108 samples** in total. At every step apply softmax to `logits / temperature`, sample one word, append it, and retain the last 20 words as context. `<unk>` is suppressed during generation; no top-k or top-p filtering is used.

Lower temperature concentrates probability on likely words; higher temperature flattens the distribution. Whether that improves readability is judged from the actual samples. Distinct-1/2 measure lexical diversity; repeated-trigram rate measures duplicated three-word phrases. None measures semantic coherence.'''),
code('''samples = pd.read_json(OUT/'samples.json')
assert len(samples) == 108
assert samples.generated.str.split().str.len().eq(100).all()
display(pd.read_csv(OUT/'temperature_summary.csv'))
display(Image(filename=str(OUT/'temperature_effects.png')))
display(Markdown((OUT/'representative_samples.md').read_text(encoding='utf-8')))'''),
md('''## 5. Qualitative analysis and conclusion
The representative samples were chosen by a fixed rule: prompt 0 and sampling seed 101, for all four models and all three temperatures. Semantic judgments are descriptive, not blinded human-study scores. The complete sample collection remains available for inspection.'''),
code('''review = OUT/'qualitative_review.md'
if review.exists():
    display(Markdown(review.read_text(encoding='utf-8')))
else:
    print('Qualitative review has not yet been written; inspect representative_samples.md.')
display(Markdown((OUT/'summary.md').read_text(encoding='utf-8')))'''),
md('''## 6. Reproducibility and limitations
Only one initialization seed is trained per architecture. Sampling is repeated, but that does not estimate training variance. A small book and a chronological split introduce limited data and topic shift. The 20-word context, lack of punctuation and unknown-word mapping limit fluent generation. Parameter counts vary by architecture. Report this run as evidence about this dataset, not proof that an architecture always wins.

Commands: `python -m unittest -v test_experiment.py`, `python experiment.py --smoke`, and `python experiment.py --full`. A full rerun updates numeric reports; the qualitative review must then be rechecked against newly generated samples. Checkpoints live in `checkpoints/full/` and are excluded from Git.''')
])
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
nbf.write(nb,ROOT/'Assignment2_Alice_Text_Generation.ipynb')
print('Notebook created.')
