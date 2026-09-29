"""Evaluate both trained models on the held-out test set.

Run:  python -m src.evaluate
Writes results/metrics.json, results/comparison.md, results/test_outputs.tsv,
results/examples.md and figures (training curves, quality by length).
"""
import json
import math
import time

import numpy as np
import pandas as pd
import sacrebleu
import torch
import torch.nn as nn

from . import config as C
from .data import ParallelData, load_split
from .models import count_params
from .plotting import plt, setup_fonts
from .train import run_epoch
from .translator import Translator

ALL_NAMES = {"seq2seq": "Seq2Seq-LSTM", "attention": "Attn-LSTM (Luong)", "bahdanau": "Attn-LSTM (Bahdanau)"}
# Only models that have been trained (checkpoint + history) are evaluated.
NAMES = {k: v for k, v in ALL_NAMES.items()
         if (C.MODEL_DIR / f"{k}.pt").exists() and (C.RESULTS_DIR / f"history_{k}.json").exists()}
BEAM_SUBSET = 1000


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def bleu(hyps, refs, tokenize="intl"):
    # "intl" splits Unicode punctuation, so the Ethiopic full stop "።" becomes its
    # own token; the default "13a" only splits ASCII punctuation.
    return round(sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize).score, 2)


def corpus_scores(hyps, refs):
    return {
        "bleu": bleu(hyps, refs),
        "bleu13a": bleu(hyps, refs, "13a"),
        "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
        "chrf++": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
    }


def evaluate_model(name, test_df):
    tr = Translator(name)
    dev = tr.dev
    hist = json.loads((C.RESULTS_DIR / f"history_{name}.json").read_text())
    res = {"model": NAMES[name], "parameters": count_params(tr.model),
           "training_time_min": round(hist["training_time_sec"] / 60, 1),
           "epochs_run": hist["epochs_run"], "best_epoch": hist["best_epoch"]}
    bench = C.RESULTS_DIR / "speed_benchmark.json"
    if bench.exists():
        b = json.loads(bench.read_text())[name]
        res["isolated_min_per_epoch"] = b["isolated_min_per_epoch"]
        res["isolated_training_time_min_estimate"] = b["isolated_training_time_min_estimate"]

    test_data = ParallelData(test_df, tr.sp_en, tr.sp_am)
    loss = run_epoch(tr.model, test_data, nn.CrossEntropyLoss(ignore_index=C.PAD), dev)
    res["test_loss"] = round(loss, 4)
    res["test_perplexity"] = round(math.exp(loss), 2)

    srcs, refs = test_df["en"].tolist(), test_df["am"].tolist()
    tr.translate_batch(srcs[:256])                             # warm-up (kernel compilation)
    sync(dev)
    t0 = time.perf_counter()
    hyps = tr.translate_batch(srcs)
    sync(dev)
    total = time.perf_counter() - t0
    res.update({f"{k}_greedy": v for k, v in corpus_scores(hyps, refs).items()})
    res["inference_time_test_set_sec"] = round(total, 2)
    res["inference_ms_per_sentence_batched"] = round(1000 * total / len(srcs), 2)

    # Single-sentence latency as experienced by the deployed API (beam = 5).
    lat, beam_hyps = [], []
    for s in srcs[:BEAM_SUBSET]:
        r = tr.translate(s, beam=5)
        lat.append(r["latency_ms"])
        beam_hyps.append(r["translation"])
    res["latency_ms_single_sentence_beam5_median"] = round(float(np.median(lat)), 1)
    greedy_sub = corpus_scores(hyps[:BEAM_SUBSET], refs[:BEAM_SUBSET])
    beam_sub = corpus_scores(beam_hyps, refs[:BEAM_SUBSET])
    res[f"bleu_greedy_first{BEAM_SUBSET}"] = greedy_sub["bleu"]
    res[f"bleu_beam5_first{BEAM_SUBSET}"] = beam_sub["bleu"]
    res[f"chrf_greedy_first{BEAM_SUBSET}"] = greedy_sub["chrf"]
    res[f"chrf_beam5_first{BEAM_SUBSET}"] = beam_sub["chrf"]
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
    return res, hyps


def by_length(df, cols):
    bins = [0, 10, 20, 30, 40, 60, 200]
    labels = ["1-10", "11-20", "21-30", "31-40", "41-60", "61+"]
    df = df.assign(bucket=pd.cut(df["en_len"], bins=bins, labels=labels))
    rows = []
    for b, g in df.groupby("bucket", observed=True):
        row = {"src_subwords": b, "sentences": len(g)}
        for c in cols:
            row[f"{c}_bleu"] = bleu(g[c].tolist(), g["am"].tolist())
            row[f"{c}_chrf"] = round(sacrebleu.corpus_chrf(g[c].tolist(), [g["am"].tolist()]).score, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_history():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, name in enumerate(NAMES):
        color = f"C{i}"                      # one color per model; dashed = train, solid = validation
        h = json.loads((C.RESULTS_DIR / f"history_{name}.json").read_text())["history"]
        ep = [x["epoch"] for x in h]
        axes[0].plot(ep, [x["train_loss"] for x in h], "--", color=color, label=f"{NAMES[name]} train")
        axes[0].plot(ep, [x["val_loss"] for x in h], "-o", color=color, ms=3, label=f"{NAMES[name]} val")
        axes[1].plot(ep, [x["val_ppl"] for x in h], "-o", color=color, ms=3, label=NAMES[name])
    axes[0].set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Validation perplexity", xlabel="epoch", ylabel="perplexity")
    for a in axes:
        a.legend()
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "training_curves.png", dpi=150)


def plot_length(tab):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(tab))
    w = .8 / len(NAMES)
    for i, name in enumerate(NAMES):
        off = (i - (len(NAMES) - 1) / 2) * w
        axes[0].bar(x + off, tab[f"{name}_bleu"], w, label=NAMES[name])
        axes[1].bar(x + off, tab[f"{name}_chrf"], w, label=NAMES[name])
    for a, t in zip(axes, ["BLEU", "chrF"]):
        a.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(tab["src_subwords"], tab["sentences"])])
        a.set(title=f"{t} by source length (subword tokens)", ylabel=t)
        a.legend()
        a.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "quality_by_length.png", dpi=150)


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    test_df = load_split("test")
    results, outputs = {}, test_df[["en", "am", "en_len", "am_len"]].copy()
    for name in NAMES:
        results[name], outputs[name] = evaluate_model(name, test_df)
    outputs.to_csv(C.RESULTS_DIR / "test_outputs.tsv", sep="\t", index=False)

    tab = by_length(outputs, list(NAMES))
    tab.to_csv(C.RESULTS_DIR / "quality_by_length.csv", index=False)
    plot_history()
    plot_length(tab)

    metrics = {"test_sentences": len(test_df), "models": results,
               "quality_by_length": tab.astype({"src_subwords": str}).to_dict("records")}
    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    rows = [
        ("BLEU (greedy, full test set, intl tok.)", "bleu_greedy"),
        ("BLEU (greedy, full test set, 13a tok.)", "bleu13a_greedy"),
        ("chrF (greedy, full test set)", "chrf_greedy"),
        ("chrF++ (greedy, full test set)", "chrf++_greedy"),
        (f"BLEU (beam 5, first {BEAM_SUBSET})", f"bleu_beam5_first{BEAM_SUBSET}"),
        (f"chrF (beam 5, first {BEAM_SUBSET})", f"chrf_beam5_first{BEAM_SUBSET}"),
        ("Test loss (cross-entropy)", "test_loss"),
        ("Test perplexity", "test_perplexity"),
        ("Training time, wall clock, concurrent runs (min)", "training_time_min"),
        ("Training time per epoch, isolated (min)", "isolated_min_per_epoch"),
        ("Training time, isolated estimate (min)", "isolated_training_time_min_estimate"),
        ("Epochs run (best epoch)", None),
        ("Inference time, whole test set, batched greedy (s)", "inference_time_test_set_sec"),
        ("Inference per sentence, batched greedy (ms)", "inference_ms_per_sentence_batched"),
        ("Latency, single sentence, beam 5, median (ms)", "latency_ms_single_sentence_beam5_median"),
        ("Trainable parameters", "parameters"),
    ]
    lines = ["| Metric | " + " | ".join(NAMES.values()) + " |", "|---|" + "---:|" * len(NAMES)]
    for label, key in rows:
        if key is None:
            vals = [f"{results[n]['epochs_run']} ({results[n]['best_epoch']})" for n in NAMES]
        elif key == "parameters":
            vals = [f"{results[n][key]:,}" for n in NAMES]
        else:
            vals = [str(results[n].get(key, "n/a")) for n in NAMES]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    (C.RESULTS_DIR / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Side-by-side examples: a spread over lengths, fixed seed for reproducibility.
    ex = outputs.sample(frac=1, random_state=C.SEED)
    picks = pd.concat([ex[ex.en_len <= 12].head(6), ex[(ex.en_len > 12) & (ex.en_len <= 25)].head(6),
                       ex[ex.en_len > 25].head(4)])
    md = ["# Translation examples (test set, greedy decoding)\n"]
    for i, r in enumerate(picks.itertuples(), 1):
        md += [f"**{i}.** Source: {r.en}  ", f"Reference: {r.am}  "]
        md += [f"{label}: {getattr(r, n)}  " for n, label in NAMES.items()]
        md[-1] += "\n"
    (C.RESULTS_DIR / "examples.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
