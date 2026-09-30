| Metric | Seq2Seq-LSTM | Attention-LSTM |
|---|---:|---:|
| BLEU (greedy, full test set, intl tok.) | 5.29 | 11.95 |
| BLEU (greedy, full test set, 13a tok.) | 4.17 | 10.14 |
| chrF (greedy, full test set) | 16.95 | 26.48 |
| chrF++ (greedy, full test set) | 15.82 | 25.28 |
| BLEU (beam 5, first 1000) | 5.39 | 12.64 |
| chrF (beam 5, first 1000) | 16.74 | 27.34 |
| Test loss (cross-entropy) | 3.5646 | 2.8403 |
| Test perplexity | 35.33 | 17.12 |
| Training time, wall clock, concurrent runs (min) | 217.9 | 312.2 |
| Training time per epoch, isolated (min) | 15.94 | 32.15 |
| Training time, isolated estimate (min) | 100.0 | 201.8 |
| Epochs | 6 + 2 fine-tuning | 6 + 2 fine-tuning |
| Inference time, whole test set, batched greedy (s) | 7.92 | 10.83 |
| Inference per sentence, batched greedy (ms) | 0.96 | 1.31 |
| Latency, single sentence, beam 5, median (ms) | 110.8 | 223.2 |
| Trainable parameters | 14,507,840 | 16,737,600 |
