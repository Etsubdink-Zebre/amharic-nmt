# English → Amharic Neural Machine Translation

Deep Learning — Group Project

| Group member | ID |
|---|---|
| Etsubdink Zebre | GSE/0523/18 |
| Franci Ayele | GSE/1254/18 |
| Henock Bonsa | GSE/3554/18 |

LSTM encoder–decoder models for English → Amharic translation: a **basic Seq2Seq-LSTM** and an
**Attention-LSTM** (Bahdanau additive attention). A Luong-attention variant is included as an ablation.
All are trained from scratch in PyTorch on 149k English–Amharic sentence pairs, evaluated on a
held-out test set, analysed for errors and attention behaviour, and deployed as a
**Streamlit** web app and a **FastAPI** REST API.

The full write-up is in [REPORT.md](REPORT.md).

---

## Results at a glance

Held-out test set, 8,815 sentences, greedy decoding:

| | Seq2Seq-LSTM | Attn-LSTM (Luong, ablation) | **Attention-LSTM (Bahdanau)** |
|---|---:|---:|---:|
| BLEU | 5.96 | 7.38 | **11.20** |
| chrF | 16.59 | 19.09 | **24.44** |
| Test loss / perplexity | 3.756 / 42.8 | 3.580 / 35.9 | **3.187 / 24.2** |
| Training time (15 epochs, isolated) | **50 min** | 109 min | 92 min |
| Inference, per sentence (batched / beam-5 API) | **0.65 / 87 ms** | 1.26 / 188 ms | 1.19 / 183 ms |
| Parameters | **14.5 M** | 16.3 M | 16.7 M |

The Bahdanau Attention-LSTM wins at every sentence length, and the gap grows on long sentences.
The Luong variant's attention collapsed onto the sentence-final token (see REPORT §4.3).

```
POST /translate  {"text": "I am going to the university."}
→ {"translation": "ወደ ዩኒቨርሲቲዬ እሄዳለሁ።", "model": "bahdanau", "latency_ms": 94}
```

Outputs: [comparison table](results/comparison.md) · [examples](results/examples.md) ·
[error examples](results/error_examples.md) · [error statistics](results/error_analysis.json) ·
[figures](results/figures/) (training curves, quality by length, 16 attention heatmaps).

---

## Project layout

```
nmt-amharic/
├── src/
│   ├── text.py          normalization shared by training and the app (single source of truth)
│   ├── prepare_data.py  download → clean → dedupe → split → SentencePiece → length filter
│   ├── data.py          tokenized datasets, length-bucketed batching
│   ├── models.py        Seq2Seq, BahdanauSeq2Seq, AttnSeq2Seq (Luong), greedy + beam decoding
│   ├── train.py         training loop, early stopping, checkpointing
│   ├── translator.py    inference pipeline (raw text → Amharic), used by eval and app
│   ├── benchmark.py     isolated training-speed benchmark
│   ├── evaluate.py      BLEU / chrF / test loss / timings / examples / plots
│   ├── analysis.py      error analysis + attention heatmaps
│   └── config.py        all paths and hyper-parameters
├── streamlit_app.py     Streamlit web app (deployed on Streamlit Community Cloud)
├── app/
│   ├── main.py          FastAPI: POST /translate, GET /health, GET / (web UI)
│   └── static/index.html
├── models/              trained checkpoints (seq2seq.pt, bahdanau.pt, attention.pt = Luong) + SentencePiece models (spm_*.model)
├── data/processed/      cleaned train/val/test TSVs
├── results/             metrics, comparison table, examples, error analysis, figures/
├── tests/               unit tests for text normalization
├── run_all.sh           reproduces everything end to end
├── requirements.txt
└── Dockerfile
```

## Installation

Python 3.10+ is required.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Running the application (uses the trained models in `models/`)

**Streamlit web app:** translation, model and beam choice, and the attention heatmap.

```bash
streamlit run streamlit_app.py
```

**FastAPI REST API:** `POST /translate`, as in the assignment brief, plus a small web UI.

```bash
uvicorn app.main:app --port 8000
```

Open http://localhost:8000 for the web UI (translation + live attention heatmap), or
http://localhost:8000/docs for the interactive API docs.

```bash
curl -X POST localhost:8000/translate -H "Content-Type: application/json" -d '{"text": "I am going to the university."}'
```

Request fields: `text` (required), `model` (`"bahdanau"` = Attention-LSTM, default; `"seq2seq"`; or `"attention"` = Luong ablation),
`beam` (1–10, default 5), `return_attention` (default `false`).
Response: `{"translation": "...", "model": "bahdanau", "latency_ms": 94}`.

### Deploying the Streamlit app

1. Sign in at https://share.streamlit.io with GitHub and click **Create app**.
2. Choose this repository, branch `main`, main file `streamlit_app.py`.
3. Under **Advanced settings**, choose Python 3.12, then **Deploy**. `requirements.txt` installs the CPU build of PyTorch.

### Docker (FastAPI)

```bash
docker build -t amharic-nmt .
```

```bash
docker run -p 8000:8000 amharic-nmt
```

## Reproducing the experiments

Everything — download, preprocessing, training both models, benchmark, evaluation and
analysis — runs with one script (≈ 4.5 h on an Apple M1 Pro GPU; CUDA is used if available):

```bash
./run_all.sh
```

Or step by step:

```bash
python -m src.prepare_data
```

```bash
python -m src.train --model seq2seq
```

```bash
python -m src.train --model bahdanau
```

```bash
python -m src.train --model attention
```

```bash
python -m src.benchmark
```

```bash
python -m src.evaluate
```

```bash
python -m src.analysis
```

Unit tests:

```bash
python -m pytest tests
```

## Dataset

[`habtew/english-amharic-translation`](https://huggingface.co/datasets/habtew/english-amharic-translation)
on the Hugging Face Hub (237,243 raw pairs). The dataset card declares **no license**;
the text is largely drawn from publicly available Bible / Jehovah's Witnesses publications
and news sites, so it is used here for non-commercial academic research only.
See [REPORT.md §1](REPORT.md#1-dataset--preprocessing) for the cleaning steps and statistics.
