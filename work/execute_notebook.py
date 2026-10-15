import os, sys, json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

workspace=Path(__file__).resolve().parent.parent
project=workspace/'outputs/alice-text-generation'
jupyter=workspace/'work/jupyter'
kernel=jupyter/'kernels/python3'
kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Python 3','language':'python'}))
os.environ['JUPYTER_PATH']=str(jupyter)
os.environ['JUPYTER_RUNTIME_DIR']=str(jupyter/'runtime')
os.environ['IPYTHONDIR']=str(jupyter/'ipython')
os.environ['MPLCONFIGDIR']=str(workspace/'work/matplotlib')
notebook=project/'Assignment2_Alice_Text_Generation.ipynb'
nb=nbformat.read(notebook,as_version=4)
client=NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(project)}})
client.execute()
nbformat.validate(nb)
nbformat.write(nb,notebook)
assert all(c.get('execution_count') is not None for c in nb.cells if c.cell_type=='code')
assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
print('Executed and validated all notebook cells in a fresh kernel.')
