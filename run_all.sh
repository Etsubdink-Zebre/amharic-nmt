#!/usr/bin/env bash
# Full pipeline: data → train all three models → isolated speed benchmark → evaluate → analysis.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p logs
$PY -m src.prepare_data
$PY -u -m src.train --model seq2seq   | tee logs/train_seq2seq.log
$PY -u -m src.train --model bahdanau  | tee logs/train_bahdanau.log
$PY -u -m src.train --model attention | tee logs/train_attention.log   # Luong ablation
$PY -m src.benchmark
$PY -m src.evaluate
$PY -m src.analysis
