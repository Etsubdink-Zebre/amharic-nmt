# English → Amharic Neural Machine Translation

Deep Learning — Group Project

| Group member | ID |
|---|---|
| Etsubdink Zebre | GSE/0523/18 |
| Franci Ayele | GSE/1254/18 |
| Henock Bonsa | GSE/3554/18 |

LSTM encoder–decoder models for English → Amharic translation: a **basic Seq2Seq-LSTM** and an
**Attention-LSTM** (Bahdanau additive attention). Both are trained from scratch in PyTorch on 692k
English–Amharic sentence pairs (habtew + OPUS MT560), evaluated on a held-out test set, analysed for
errors and attention behaviour, and deployed as a **Streamlit** web app and a **FastAPI** REST API.

**Live app:** https://amharic-nmt.streamlit.app · **Full write-up:** [REPORT.md](REPORT.md)

---

## Results at a glance

Held-out test set: 8,266 sentences, greedy decoding.

| | Seq2Seq-LSTM | **Attention-LSTM** |
|---|---:|---:|
| BLEU | 5.29 | **11.95** |
| chrF | 16.95 | **26.48** |
| Test loss / perplexity | 3.565 / 35.3 | **2.840 / 17.1** |
| Training time (6 + 2 fine-tuning epochs, isolated) | **100 min** | 202 min |
| Inference per sentence (batched / beam-5 app) | **0.96 / 111 ms** | 1.31 / 223 ms |
| Parameters | **14.5 M** | 16.7 M |

The Attention-LSTM wins at every sentence length, and the gap grows on long sentences.
A Luong-attention variant tried first collapsed onto the sentence-final token (REPORT §4.3).

```
POST /translate  {"text": "I am going to the university."}
→ {"translation": "ወደ ዩኒቨርሲቲ ገብቼ እሄዳለሁ።", "model": "bahdanau", "latency_ms": 223, "warnings": []}
```

Outputs: [comparison table](results/comparison.md) · [examples](results/examples.md) ·
[error examples](results/error_examples.md) · [error statistics](results/error_analysis.json) ·
[figures](results/figures/) (training curves, quality by length, attention heatmaps) ·
earlier phases: [phase-1 results](results/phase1_comparison.md), [phase-1 scores on this test set](results/phase1_scores.json),
[before fine-tuning](results/phase2_pre_ft_scores.json), [without repetition blocking](results/no_repeat_blocking_scores.json).

---

## Project layout

```
nmt-amharic/
├── src/
│   ├── text.py          normalization shared by training and the app (single source of truth)
│   ├── prepare_data.py  download → clean → dedupe → split → SentencePiece → length filter
│   ├── data.py          tokenized datasets, length-bucketed batching
│   ├── models.py        Seq2Seq, BahdanauSeq2Seq, AttnSeq2Seq (Luong, ablation), greedy + beam decoding
│   ├── train.py         training loop, early stopping, checkpointing
│   ├── finetune.py      in-domain fine-tuning on the habtew training pairs
│   ├── eval_phase1.py   scores archived phase-1 checkpoints on the current test set
│   ├── translator.py    inference pipeline (raw text → Amharic), used by eval and app
│   ├── benchmark.py     isolated training-speed benchmark
│   ├── evaluate.py      BLEU / chrF / test loss / timings / examples / plots
│   ├── analysis.py      error analysis + attention heatmaps
│   └── config.py        all paths and hyper-parameters
├── streamlit_app.py     Streamlit web app (deployed on Streamlit Community Cloud)
├── app/
│   ├── main.py          FastAPI: POST /translate, GET /health, GET / (web UI)
│   └── static/index.html
├── models/              trained checkpoints (seq2seq.pt, bahdanau.pt), SentencePiece models, training word counts
├── data/                created by prepare_data (not in the repository)
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

Request fields: `text` (required), `model` (`"bahdanau"` = Attention-LSTM, default; or `"seq2seq"`),
`beam` (1–10, default 5), `return_attention` (default `false`).
Response: `{"translation": "...", "model": "bahdanau", "latency_ms": 223, "warnings": [...]}`.

### Translation scope

The models only know what their training data contains: about 692k full sentences, mostly from
religious and news writing.

| Works well | Unreliable |
|---|---|
| Complete sentences of about 5–30 words | Single words and short phrases ("Bye", "Hello") |
| Everyday topics: family, work, places, society, news, religion | Greetings, chat and slang ("lol", "ok") |
| Common names (Jesus, Moses, Ethiopia, Addis Ababa) | Rare names, technical and scientific terms |
| | Very long sentences (over ~50 words) |

Both apps show this note. Each translation also comes with **warnings** when the input is shorter than
4 words, has no final punctuation, longer than 50 words, or contains English words that appeared fewer than 30 times
(or never) in the training data. The counts are in `models/en_word_freq.json`. The API returns
them in a `warnings` list.

**Spelling suggestions.** When a word never appears in the training data but is one or two typos
away from a common training word, the apps suggest the fix ("switherland" → "switzerland",
"hospitl" → "hospital"). Streamlit offers a one-click "Did you mean …? Translate that instead" button,
and the API returns the fixes in a `suggestions` object. Distance is Damerau–Levenshtein, so a swapped
pair of letters counts as one typo ("thnak" → "thank").

### Deploying the Streamlit app

The app is live at https://amharic-nmt.streamlit.app and redeploys on every push to `main`. To deploy your own copy:

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
analysis — runs with one script (≈ 5.5 h on an Apple M1 Pro GPU; CUDA is used if available):

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
python -m src.finetune --model seq2seq
```

```bash
python -m src.finetune --model bahdanau
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

## Datasets

* [`habtew/english-amharic-translation`](https://huggingface.co/datasets/habtew/english-amharic-translation)
  (237,243 raw pairs): training, validation and test. The dataset card declares **no license**. The text
  is largely drawn from publicly available Bible / Jehovah's Witnesses publications and news sites,
  so it is used here for non-commercial academic research only and is not redistributed.
* [`michsethowusu/english-amharic_sentence-pairs_mt560`](https://huggingface.co/datasets/michsethowusu/english-amharic_sentence-pairs_mt560)
  (OPUS MT560, 669,145 pairs, **CC-BY-4.0**): additional training data only.

`python -m src.prepare_data` downloads both. See [REPORT.md §1](REPORT.md#1-dataset--preprocessing)
for the cleaning steps, the leakage we found and removed, and the statistics.
