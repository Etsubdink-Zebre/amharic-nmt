#!/usr/bin/env bash
# Full pipeline: data → train both models → in-domain fine-tuning → speed benchmark → evaluate → analysis.
# (The phase-1 Luong ablation is described in REPORT.md; train it with `src.train --model attention`
#  on the habtew-only data if needed.)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p logs
$PY -m src.prepare_data
$PY -u -m src.train --model seq2seq   | tee logs/train_seq2seq.log
$PY -u -m src.train --model bahdanau  | tee logs/train_bahdanau.log
$PY -u -m src.finetune --model seq2seq  | tee logs/ft_seq2seq.log
$PY -u -m src.finetune --model bahdanau | tee logs/ft_bahdanau.log
$PY -m src.benchmark
$PY -m src.evaluate
$PY -m src.analysis
