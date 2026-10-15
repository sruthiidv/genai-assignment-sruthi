# Qualitative review and final conclusion

## Selection and assessment method

Review the fixed representative sample for **prompt 0, sampling seed 101** at each model/temperature combination. The entire generated continuation is retained in `representative_samples.md`; the complete 108-sample collection is in `samples.json` and `samples.csv`. These are descriptive, AI-assisted observations of the recorded outputs, not independent human ratings or evidence of statistical significance. The student should inspect and understand these observations before submission.

Coherence means meaningful, grammatically connected statements. Contextual consistency means continuing the prompt and maintaining subjects/events, rather than merely mentioning characters from the book. Diversity and repetition are assessed alongside the nine-sample quantitative averages for each configuration. Punctuation was removed from the training representation, so the absence of sentence punctuation is expected.

## Fixed-sample comparison

| Model | Temperature | Coherence and contextual consistency | Diversity and repetition |
| --- | --- | --- | --- |
| Simple RNN | 0.5 | Recognizable short phrases such as “looked at once” and “the right size”, but subjects switch abruptly and clauses do not form a sustained scene. The opening fails to complete the prompt's “nothing to”. | Common phrases such as “a little” and “she had” recur; limited variation. |
| Simple RNN | 1.0 | More varied story words, including the Cat and Mock Turtle, but constructions such as “the queen voice” remain ungrammatical and the original scene is lost. | Greater variation and few exact repeated trigrams; grammatical inconsistency remains. |
| Simple RNN | 1.5 | Fragments such as “executioner replied the queen pardon” and “savage nicely delight” have weak semantic connections. | Broad word variety is mostly noise rather than useful story development. |
| Bidirectional RNN | 0.5 | Some familiar constructions, including “in a minute or two”, but confused subject relations and no sustained continuation of the opening scene. | Particularly frequent references to “the queen”; strongest low-temperature lexical concentration among the models. |
| Bidirectional RNN | 1.0 | Story entities are present, but “the girls who will the often” illustrates broken syntax and unstable meaning. | Variety increases substantially, with very little repeated-trigram content on average. |
| Bidirectional RNN | 1.5 | Word combinations such as “walk eyes corner” are hard to interpret. Long-range narrative consistency is absent. | High diversity and zero repeated trigrams on average do not imply readable output. |
| LSTM | 0.5 | Short constructions such as “a good deal frightened” and “in a minute or two” are plausible, but “a large little thing” and “two two” show inconsistency. It does not sustain the opening scene. | Repeats variations of “she had”, “little thing” and “good deal”; lexical variety is restricted. |
| LSTM | 1.0 | Contains meaningful local phrases such as “no live cats” but rapidly shifts among unrelated subjects. Overall coherence remains weak. | More varied than 0.5 with negligible repeated-trigram content; still not a coherent narrative. |
| LSTM | 1.5 | Expressions such as “have wasting gloves” and “advisable interesting” have poor grammatical and semantic compatibility. | Highest average distinct-1 of the configurations, but that diversity comes with poor readability. |
| GRU | 0.5 | Relatively plausible local actions, including “she went back to the door and hurried off”, and recurring story motifs such as the golden key. Still switches context and does not form a consistent scene. | Repeats “to herself” and “little golden key”, although average distinct-1/2 is higher than the other models at 0.5. |
| GRU | 1.0 | Some recognizable fragments such as “in the middle” are surrounded by incompatible clauses, for example “she had executioner”. | More vocabulary variation, but no clear gain in sustained coherence. |
| GRU | 1.5 | Fragments such as “eyes no often listening savage” have little interpretable structure. | Broad lexical exploration with almost no phrase repetition, but low usefulness as prose. |

## What temperature changed

Across all four models, increasing temperature raised average distinct-1 and distinct-2 and reduced repeated-trigram rate. At 0.5, mean distinct-1 ranges from **0.4467 to 0.5478** and repeated-trigram rate from **0.0385 to 0.0658**. At 1.0, mean distinct-1 rises to **0.7144–0.7467**, while repetition is **0–0.0011**. At 1.5, distinct-1 reaches **0.8778–0.9056**, and all configurations have distinct-2 of 1.0 and repeated-trigram rate of zero.

These are averages over nine samples per configuration, not scores for just the representative passage. For 100-word samples, distinct-2 readily saturates at 1.0; this means no bigram repeats within a sample, not that output quality is perfect. Raising temperature clearly improves lexical diversity but does not improve semantic coherence here. Among the tested values, **0.5 is the most defensible choice for locally readable fragments**, accepting its repetition; 1.0 offers greater variety, and 1.5 is too disjointed for this corpus and training setup.

## Which model is most effective?

**Choose LSTM as the overall model for this experiment**, primarily because it achieves the lowest test cross-entropy (**4.5904**), lowest perplexity (**98.5352**) and highest top-1 accuracy (**18.8494%**). It improves perplexity by approximately **6.85%** relative to Simple RNN. Its training time was **28.18 seconds**, compared with **13.52 seconds** for Simple RNN, **21.36 seconds** for Bidirectional RNN and **34.22 seconds** for GRU on the recorded CPU setup.

The qualitative evidence does not show an unambiguous LSTM advantage in fluent generation: **GRU at 0.5 has some of the more plausible local phrases and the best low-temperature diversity/repetition balance**. LSTM is therefore the best measured next-word model here, while the evidence for a clear semantic-generation winner is limited. Simple RNN is the fastest baseline. Bidirectional RNN has the largest parameter count but the weakest held-out results in this run; bidirectionality over the known context does not guarantee improved prediction.

All models overfit: training loss continues to fall after validation loss reaches its minimum. Best epochs were Simple RNN 6, Bidirectional RNN 3, LSTM 7 and GRU 6. Early stopping retained those checkpoints rather than the final epoch weights.

## Limits and potential improvements

The corpus contains **26,615 words**, and the training-only vocabulary has **1,291 entries including `<unk>`**. Approximately **11.69% of validation words and 12.81% of test words** map to `<unk>`, compared with **4.80% of training words**. Reported loss and accuracy include these unknown targets. This limits comparisons to systems using other vocabulary policies and partly reflects chronological vocabulary/topic shift.

The models see only 20 preceding words and no punctuation. A single training seed cannot establish stable architecture rankings, and the qualitative review is neither blinded nor a human-subject evaluation. More training data, regularization, several training seeds and carefully controlled context-length experiments would strengthen future work. These are proposed extensions, not unperformed experiments presented as results.
