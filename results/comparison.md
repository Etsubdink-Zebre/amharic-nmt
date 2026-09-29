| Metric | Seq2Seq-LSTM | Attn-LSTM (Luong) | Attn-LSTM (Bahdanau) |
|---|---:|---:|---:|
| BLEU (greedy, full test set, intl tok.) | 5.96 | 7.38 | 11.2 |
| BLEU (greedy, full test set, 13a tok.) | 4.76 | 5.96 | 9.31 |
| chrF (greedy, full test set) | 16.59 | 19.09 | 24.44 |
| chrF++ (greedy, full test set) | 15.55 | 18.07 | 23.33 |
| BLEU (beam 5, first 1000) | 6.18 | 7.72 | 12.08 |
| chrF (beam 5, first 1000) | 16.87 | 19.16 | 25.42 |
| Test loss (cross-entropy) | 3.7563 | 3.5796 | 3.1872 |
| Test perplexity | 42.79 | 35.86 | 24.22 |
| Training time, wall clock, concurrent runs (min) | 105.6 | 164.8 | 107.9 |
| Training time per epoch, isolated (min) | 3.32 | 7.26 | 6.16 |
| Training time, isolated estimate (min) | 49.8 | 108.9 | 92.4 |
| Epochs run (best epoch) | 15 (15) | 15 (15) | 15 (15) |
| Inference time, whole test set, batched greedy (s) | 5.69 | 11.1 | 10.5 |
| Inference per sentence, batched greedy (ms) | 0.65 | 1.26 | 1.19 |
| Latency, single sentence, beam 5, median (ms) | 86.7 | 187.6 | 182.6 |
| Trainable parameters | 14,507,840 | 16,343,360 | 16,737,600 |
