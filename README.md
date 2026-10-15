# Alice in Wonderland: Comparing Four Autoregressive Text Generators

Foundations of Generative AI (ICT4403) — Assignment 2.

## Objective

Train Simple RNN, Bidirectional RNN, LSTM and GRU models from scratch to predict the next word and generate text sequentially. Compare held-out loss, perplexity, accuracy, training cost, generated-text quality and sampling temperature.

## Dataset

[Alice's Adventures in Wonderland, Lewis Carroll — Project Gutenberg ebook 11](https://www.gutenberg.org/ebooks/11). The raw UTF-8 text is retained in `data/alice_raw.txt`, including the Gutenberg licence. `data/provenance.json` records the download source, retrieval timestamp and SHA-256 checksum; `data/statistics.json` records measured corpus statistics.

Preprocessing removes source boilerplate, contents, chapter headings and illustration markers; normalizes Unicode punctuation; and extracts lowercase words with internal apostrophes. Punctuation is not predicted. Split the ordered words 80%/10%/10% before creating 20-word input / next-word target examples with stride one. Fit the vocabulary to training words with frequency at least two; map remaining words to `<unk>`. No window crosses a split boundary.

## Models and training

All models use a 64-dimensional embedding, one recurrent layer with hidden size 128 per direction, and a linear vocabulary head. Simple RNN uses tanh; LSTM and GRU use gated recurrence. Bidirectional RNN concatenates the final forward/backward states of the known input window; it never sees the target word. Parameter counts are reported rather than assumed equal.

Common settings: Adam (0.001), batch size 64, gradient clipping at norm 1, seed 42, up to 25 epochs, validation-loss early stopping with patience four, and best-validation checkpoint selection. The implementation uses CUDA if available and otherwise CPU, with at most four CPU threads. There is one training run per model; identical budgets can yield different stopping epochs.

## Windows setup and execution

Use Python 3.12. Open a terminal in this project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v test_experiment.py
.\.venv\Scripts\python.exe experiment.py --smoke
.\.venv\Scripts\python.exe experiment.py --full
```

Open `Assignment2_Alice_Text_Generation.ipynb` in VS Code or a Jupyter frontend using this environment. The notebook loads saved results by default; set `RETRAIN=True` to repeat training. It calls the same implementation as the command line. `python build_notebook.py` regenerates the notebook source and clears its outputs, so it is not needed to view the delivered executed notebook.

`requirements.txt` specifies compatible package ranges. `requirements-lock.txt`, when present, records the exact versions used. `results/full/environment.json` records the actual execution environment. If Windows Controlled folder access blocks Python from writing to Documents, allow the specific trusted Python executable through Windows Security or use a writable project location.

## Evaluation and results

The full experiment writes `results/full/metrics.csv`, `histories.json`, `summary.md`, and comparison plots. Test loss is example-weighted cross-entropy; perplexity is its exponential; accuracy is top-1 correctness. Unknown targets remain in evaluation. Training time includes validation and checkpoint saving, but excludes test evaluation and generation. The test set is not used for model checkpoint selection.

See [recorded results and quantitative conclusion](results/full/summary.md) and [qualitative review](results/full/qualitative_review.md). The executed notebook displays the same tables and figures.

| Model | Test loss | Perplexity | Accuracy | Training seconds | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple RNN | 4.6614 | 105.7794 | 16.8433% | 13.52 | 6 |
| Bidirectional RNN | 4.7446 | 114.9675 | 16.6162% | 21.36 | 3 |
| LSTM | 4.5904 | 98.5352 | 18.8494% | 28.18 | 7 |
| GRU | 4.6396 | 103.4997 | 18.0167% | 34.22 | 6 |

**Conclusion:** LSTM is the strongest measured next-word model in this run, with the lowest perplexity and highest accuracy. GRU at temperature 0.5 produces some relatively plausible local phrases, but none of the models generates a consistently coherent narrative. Higher temperatures increase lexical diversity while weakening readability. Temperature 0.5 is the preferred tested setting for readable fragments, with repetition as the tradeoff. These conclusions concern one training seed and this dataset only.

Measured dataset size: 26,615 words; vocabulary: 1,291 entries; training/validation/test examples: 21,272 / 2,641 / 2,642. Test unknown-word rate is 12.81%, and unknown targets remain in the reported metrics.

![Learning curves](results/full/learning_curves.png)
![Model comparison](results/full/model_comparison.png)

## Sampling-temperature analysis

Generate 100 new words from each of three fixed 20-word training passages, at temperatures 0.5, 1.0 and 1.5, with sampling seeds 101, 202 and 303: 108 samples. Use categorical sampling from `softmax(logits / temperature)`, suppress `<unk>`, and retain the last 20 words. No top-k/top-p filtering is applied.

All samples are retained in `results/full/samples.json` and `samples.csv`. `temperature_summary.csv` averages distinct-1, distinct-2 and repeated-trigram rate over nine samples per model/temperature. These metrics use generated words only. Higher lexical diversity is not necessarily better semantic coherence. Fixed representative examples use prompt 0 and sampling seed 101, avoiding cherry-picking.

![Temperature effects](results/full/temperature_effects.png)

## Reproduction, limitations and repository contents

- `experiment.py`: data preparation, architectures, training, evaluation and sampling.
- `reporting.py`: measured tables, plots and representative samples.
- `test_experiment.py`: split/vocabulary checks, gradient updates, checkpoint fidelity, weighted evaluation and generation checks.
- `data/`: raw and processed corpus, provenance, vocabulary, statistics and fixed seeds.
- `results/full/`: complete experimental results, plots and analysis.
- `checkpoints/full/`: best local checkpoints, deliberately ignored by Git.

A small single-book corpus, chronological topic shift, unknown-word mapping, removal of punctuation and a 20-word context constrain quality. One training seed does not establish statistical superiority. Equal hidden sizes do not equalize parameter counts. Rerunning training requires rechecking any manually written qualitative judgments. Smoke outputs are isolated and must not be cited as full experiment results.

## Submission

Repository: https://github.com/sruthiidv/genai-assignment-sruthi. Ensure evaluator access. Keep checkpoints and environment folders out of Git; include the executed notebook, data provenance, code, plots, tables, samples and this README. Provide your real student name and registration number to produce `YourName_RegNum_FGAI_Assignment2.docx` and its PDF counterpart. The submission document must also include the assignment title, system title, dataset name and dataset link. Upload that final PDF to LMS. No repository URL or personal information is fabricated in this package.
