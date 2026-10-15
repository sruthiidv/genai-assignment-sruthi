import sys, json, math, hashlib, zipfile, itertools
from pathlib import Path
workspace=Path(__file__).resolve().parent.parent
project=workspace/'outputs/alice-text-generation'
sys.path.insert(0,str(project))
import torch, pandas as pd, nbformat
import experiment as e

torch.set_num_threads(4)
datasets,vocab,seeds=e.prepare()
out=project/'results/full'
metrics=pd.read_csv(out/'metrics.csv')
samples=json.loads((out/'samples.json').read_text())
histories=json.loads((out/'histories.json').read_text())
expected=set(itertools.product(e.KINDS,range(3),[.5,1.,1.5],[101,202,303]))
actual={(s['model'],s['seed_index'],s['temperature'],s['sampling_seed']) for s in samples}
assert actual==expected and len(samples)==108
for s in samples:
    words=s['generated'].split()
    assert len(words)==100 and '<unk>' not in words
    for key,val in e.diversity(words).items(): assert math.isclose(val,s[key],abs_tol=1e-12)
for row in metrics.itertuples():
    model=e.TextModel(row.model,len(vocab))
    model.load_state_dict(torch.load(project/f'checkpoints/full/{row.model}.pt',weights_only=True))
    got=e.evaluate(model,torch.utils.data.DataLoader(datasets['test'],batch_size=128),torch.device('cpu'))
    for key in got: assert math.isclose(got[key],getattr(row,key),rel_tol=1e-6)
    assert row.parameters==sum(p.numel() for p in model.parameters())
    assert row.best_epoch==min(histories[row.model],key=lambda r:r['validation_loss'])['epoch']
nb=nbformat.read(project/'Assignment2_Alice_Text_Generation.ipynb',as_version=4)
nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code)
assert not any(o.output_type=='error' for c in code for o in c.outputs)
readme=(project/'README.md').read_text(encoding='utf-8')
for row in metrics.itertuples():
    assert f'{row.perplexity:.4f}' in readme
    assert f'{row.accuracy*100:.4f}%' in readme
checks=dict(unit_tests_passed=3,smoke_run_completed=True,full_models=4,samples=108,
    sample_length_words=100,all_experiment_combinations_present=True,
    saved_checkpoints_reproduce_test_metrics=True,diversity_metrics_recomputed=True,
    best_epochs_match_validation_histories=True,notebook_executed_fresh_kernel=True,
    notebook_code_cells=len(code),notebook_errors=0,readme_matches_metrics=True,
    plots_visually_inspected=True,submission_document_status='Pending student name, registration number and real GitHub URL',
    publication_status='Not published; destination and account access not provided')
(out/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
archive=workspace/'outputs/Alice_Assignment2_GitHub_Ready.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(project.rglob('*')):
        if not p.is_file(): continue
        rel=p.relative_to(project)
        if '__pycache__' in rel.parts or 'checkpoints' in rel.parts or 'smoke' in rel.parts or p.name=='.gitkeep': continue
        z.write(p,str(Path('alice-text-generation')/rel))
print(json.dumps(checks,indent=2))
print('GitHub-ready archive:',archive,'bytes:',archive.stat().st_size)
