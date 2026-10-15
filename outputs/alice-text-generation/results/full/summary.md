# Recorded experiment results

| model | parameters | training_seconds | epochs | best_epoch | loss | perplexity | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SimpleRNN | 273995 | 13.5247 | 10 | 6 | 4.6614 | 105.7794 | 0.1684 |
| BidirectionalRNN | 464075 | 21.3577 | 7 | 3 | 4.7446 | 114.9675 | 0.1662 |
| LSTM | 348491 | 28.1752 | 11 | 7 | 4.5904 | 98.5352 | 0.1885 |
| GRU | 323659 | 34.2181 | 10 | 6 | 4.6396 | 103.4997 | 0.1802 |

The lowest test perplexity in this run belongs to **LSTM** (98.54), with test accuracy 18.85%. This is the quantitative winner for this dataset and split, not a claim of universal superiority.

## Temperature statistics

Each row averages nine samples: three prompts × three sampling seeds. Diversity/repetition are measured on the 100 generated words only, excluding the prompt.

| model | temperature | distinct_1 | distinct_2 | repeated_trigram_rate |
| --- | --- | --- | --- | --- |
| BidirectionalRNN | 0.5000 | 0.4467 | 0.7856 | 0.0658 |
| BidirectionalRNN | 1.0000 | 0.7167 | 0.9832 | 0.0011 |
| BidirectionalRNN | 1.5000 | 0.8778 | 1.0000 | 0.0000 |
| GRU | 0.5000 | 0.5478 | 0.8687 | 0.0385 |
| GRU | 1.0000 | 0.7289 | 0.9832 | 0.0000 |
| GRU | 1.5000 | 0.8811 | 1.0000 | 0.0000 |
| LSTM | 0.5000 | 0.4889 | 0.8081 | 0.0567 |
| LSTM | 1.0000 | 0.7467 | 0.9877 | 0.0000 |
| LSTM | 1.5000 | 0.9056 | 1.0000 | 0.0000 |
| SimpleRNN | 0.5000 | 0.4989 | 0.8092 | 0.0658 |
| SimpleRNN | 1.0000 | 0.7144 | 0.9675 | 0.0000 |
| SimpleRNN | 1.5000 | 0.8833 | 1.0000 | 0.0000 |

Distinct-1 and distinct-2 are unique unigram/bigram counts divided by their total counts. Repeated-trigram rate is one minus unique trigrams divided by total trigrams. These describe lexical behaviour, not semantic coherence. High diversity can also reflect incoherent word choices.

## Interpretation limits

Only one model-initialization seed was trained. A small single-book corpus, chronological topic shifts, a fixed 20-word context, removal of punctuation, and vocabulary mapping to `<unk>` limit interpretation. Perplexity includes unknown targets, so it is not directly comparable to a different vocabulary. Parameter counts differ even with equal hidden sizes. Early stopping means different actual training durations under the same budget. Training time includes validation and checkpoint writes but excludes generation and test evaluation.

See qualitative_review.md for the fixed representative samples and semantic assessment. All 108 samples are preserved in samples.json and samples.csv.
