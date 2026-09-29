"""Error analysis for every trained model and attention visualisation for the attention models.

Run after src.evaluate:  python -m src.analysis
Writes results/error_analysis.json, results/error_examples.md,
results/attention_stats.json and results/figures/attention_*.png.

The error categories are detected with transparent heuristics (documented next
to each function) so every number in the report can be reproduced; they are
indicators, not a replacement for the manual reading summarised in the report.
"""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import sacrebleu

from . import config as C
from .data import load_split
from .evaluate import NAMES as MODELS
from .plotting import plt, setup_fonts
from .translator import Translator

WORD = re.compile(r"[^\s።፣፤?!.,:;\"'()]+")


def words(s):
    return WORD.findall(s)


# ---- heuristics ---------------------------------------------------------------
def has_repetition(hyp, ref):
    """A word repeated back-to-back, or a 2-gram occurring twice, that the reference does not have."""
    def reps(ws):
        imm = sum(a == b for a, b in zip(ws, ws[1:]))
        bi = Counter(zip(ws, ws[1:]))
        return imm + sum(c - 1 for c in bi.values() if c > 1)
    return reps(words(hyp)) > reps(words(ref))


def length_ratio(hyp, ref):
    return len(words(hyp)) / max(1, len(words(ref)))


def order_discordance(hyp, ref):
    """Among words that occur exactly once in both hypothesis and reference, the
    fraction of word pairs whose relative order differs (0 = same order)."""
    h, r = words(hyp), words(ref)
    hc, rc = Counter(h), Counter(r)
    common = [w for w in h if hc[w] == 1 and rc.get(w) == 1]
    if len(common) < 3:
        return None
    rpos = [r.index(w) for w in common]
    pairs = [(i, j) for i in range(len(rpos)) for j in range(i + 1, len(rpos))]
    return sum(rpos[i] > rpos[j] for i, j in pairs) / len(pairs)


def morph_near_misses(hyp, ref):
    """Hypothesis words that are not in the reference but share their first two
    syllables (Ethiopic characters) with a reference word — i.e. the right stem
    with the wrong affixes. Returns (near misses, hypothesis words)."""
    h, r = words(hyp), set(words(ref))
    prefixes = {w[:2] for w in r if len(w) >= 3}
    miss = [w for w in h if w not in r and len(w) >= 3 and w[:2] in prefixes]
    return len(miss), len(h)


def numbers_preserved(src, hyp):
    nums = re.findall(r"\d+", src)
    if not nums:
        return None
    return all(n in re.findall(r"\d+", hyp) for n in nums)


def proper_noun_lexicon():
    """English words that are capitalised in ≥ 90 % of their mid-sentence
    occurrences in the raw training corpus (≥ 3 occurrences) ≈ named entities."""
    raw = pd.read_parquet(C.RAW_DIR / "train.parquet")["translation"].map(lambda x: x["en"])
    cap, tot = Counter(), Counter()
    for s in raw:
        toks = re.findall(r"[A-Za-z][a-z]+", s)
        for t in toks[1:]:
            lw = t.lower()
            tot[lw] += 1
            cap[lw] += t[0].isupper()
    return {w for w, n in tot.items() if n >= 3 and cap[w] / n >= 0.9}


def chrf(hyps, refs):
    return round(sacrebleu.corpus_chrf(list(hyps), [list(refs)]).score, 2) if len(hyps) else None


def error_analysis(out, train_df):
    en_freq = Counter(w for s in train_df["en"] for w in re.findall(r"[a-z]+", s))
    names = proper_noun_lexicon()
    out = out.copy()
    out["rare"] = out["en"].map(lambda s: any(en_freq[w] <= 2 for w in re.findall(r"[a-z]+", s)))
    out["unseen"] = out["en"].map(lambda s: any(en_freq[w] == 0 for w in re.findall(r"[a-z]+", s)))
    out["has_ne"] = out["en"].map(lambda s: any(w in names for w in re.findall(r"[a-z]+", s)))
    out["long"] = out["en_len"] > 40

    report = {"subsets": {
        "sentences": len(out),
        "with_rare_source_word(freq<=2)": int(out.rare.sum()),
        "with_unseen_source_word": int(out.unseen.sum()),
        "with_named_entity": int(out.has_ne.sum()),
        "long(>40 subwords)": int(out.long.sum()),
    }}
    examples = {}
    for m in MODELS:
        hyp = out[m]
        rep = [has_repetition(h, r) for h, r in zip(hyp, out.am)]
        lr = np.array([length_ratio(h, r) for h, r in zip(hyp, out.am)])
        disc = [d for d in (order_discordance(h, r) for h, r in zip(hyp, out.am)) if d is not None]
        nm = [morph_near_misses(h, r) for h, r in zip(hyp, out.am)]
        num = [x for x in (numbers_preserved(s, h) for s, h in zip(out.en, hyp)) if x is not None]
        report[m] = {
            "repeated_words_%": round(100 * np.mean(rep), 2),
            "missing_words_%(len<0.7*ref)": round(100 * np.mean(lr < 0.7), 2),
            "additional_words_%(len>1.4*ref)": round(100 * np.mean(lr > 1.4), 2),
            "mean_length_ratio(hyp/ref words)": round(float(lr.mean()), 3),
            "word_order_discordance_mean": round(float(np.mean(disc)), 3),
            "word_order_sentences_measured": len(disc),
            "morphology_near_miss_%of_hyp_words": round(100 * sum(a for a, _ in nm) / max(1, sum(b for _, b in nm)), 2),
            "numbers_copied_correctly_%": round(100 * np.mean(num), 2),
            "chrf_all": chrf(hyp, out.am),
            "chrf_with_named_entity": chrf(hyp[out.has_ne], out.am[out.has_ne]),
            "chrf_without_named_entity": chrf(hyp[~out.has_ne], out.am[~out.has_ne]),
            "chrf_with_rare_word": chrf(hyp[out.rare], out.am[out.rare]),
            "chrf_without_rare_word": chrf(hyp[~out.rare], out.am[~out.rare]),
            "chrf_long(>40)": chrf(hyp[out.long], out.am[out.long]),
            "chrf_short(<=40)": chrf(hyp[~out.long], out.am[~out.long]),
        }
        out[f"{m}_rep"], out[f"{m}_lr"] = rep, lr

    # Example sentences per category (short enough to read, deterministic order).
    pool = out[out.en_len <= 30].sample(frac=1, random_state=C.SEED)
    any_ = lambda f: np.logical_or.reduce([f(m) for m in MODELS])
    examples["Repeated words"] = pool[any_(lambda m: pool[f"{m}_rep"])].head(4)
    examples["Missing words (under-translation)"] = pool[any_(lambda m: pool[f"{m}_lr"] < 0.6)].head(4)
    examples["Additional words (over-translation)"] = pool[any_(lambda m: pool[f"{m}_lr"] > 1.5)].head(3)
    examples["Named entities"] = pool[pool.has_ne].head(5)
    examples["Unknown / rare words"] = pool[pool.unseen].head(4)
    examples["Numbers"] = pool[pool.en.str.contains(r"\d")].head(3)
    examples["Long sentences"] = out[out.en_len > 45].sample(frac=1, random_state=C.SEED).head(3)
    return report, examples


# ---- attention -------------------------------------------------------------------
ATTN_SENTENCES = [
    "i am going to the university.",
    "the children are playing in the garden.",
    "she did not eat bread yesterday.",
    "jesus said to his disciples: love one another.",
]


def plot_attention(r, path, title):
    att = np.array(r["attention"])
    src, tgt = r["src_tokens"], r["tgt_tokens"]
    fig, ax = plt.subplots(figsize=(max(5, .45 * len(src) + 2), max(4, .4 * len(tgt) + 1.5)))
    ax.imshow(att, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(src)), src, rotation=60, ha="right", fontsize=10)
    ax.set_yticks(range(len(tgt)), tgt, fontsize=11)
    for i in range(att.shape[0]):
        for j in range(att.shape[1]):
            if att[i, j] > .3:
                ax.text(j, i, f"{att[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if att[i, j] > .6 else "black")
    ax.set_xlabel("English source (SentencePiece)")
    ax.set_ylabel("Amharic output")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def attention_study(test_df, name):
    tr = Translator(name)
    picked = []
    # custom sentences + test sentences of moderate length
    cand = test_df[(test_df.en_len >= 8) & (test_df.en_len <= 16)].sample(frac=1, random_state=7)
    items = [(s, None) for s in ATTN_SENTENCES] + list(zip(cand.en.head(4), cand.am.head(4)))
    for k, (s, ref) in enumerate(items, 1):
        r = tr.translate(s)
        plot_attention(r, C.FIG_DIR / f"attention_{name}_{k}.png", f"{MODELS[name]}: {s}\n→ {r['translation']}")
        att = np.array(r["attention"])
        picked.append({"id": k, "source": s, "reference": ref, "translation": r["translation"],
                       "figure": f"figures/attention_{name}_{k}.png",
                       "top_source_token_per_output": [
                           [t, r["src_tokens"][int(att[i].argmax())], round(float(att[i].max()), 2)]
                           for i, t in enumerate(r["tgt_tokens"])]})

    # Corpus-level attention statistics on 300 test sentences:
    # entropy (how peaked) and how monotone the argmax alignment is.
    ent, mono, rel_pos = [], [], []
    for s in test_df.en.head(300):
        r = tr.translate(s)
        att = np.array(r["attention"])[:-1]            # drop the </s> row
        if len(att) < 2:
            continue
        ent.append(float(np.mean(-(att * np.log(att + 1e-9)).sum(1))))
        am = att.argmax(1)
        mono.append(float(np.mean(np.diff(am) >= 0)))
        rel_pos += list(zip(np.linspace(0, 1, len(am)), am / max(1, att.shape[1] - 1)))
    stats = {"sentences": len(ent), "mean_attention_entropy_nats": round(float(np.mean(ent)), 3),
             "fraction_monotone_steps": round(float(np.mean(mono)), 3)}

    rp = np.array(rel_pos)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    h = ax.hist2d(rp[:, 1], rp[:, 0], bins=20, cmap="Blues")
    fig.colorbar(h[3], ax=ax, label="count")
    ax.set(xlabel="relative position of most-attended English token",
           ylabel="relative position of generated Amharic token",
           title=f"{MODELS[name]}: where the decoder looks")
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"attention_alignment_density_{name}.png", dpi=150)
    plt.close(fig)
    return picked, stats


def main():
    setup_fonts()
    out = pd.read_csv(C.RESULTS_DIR / "test_outputs.tsv", sep="\t", keep_default_na=False)
    report, examples = error_analysis(out, load_split("train"))
    (C.RESULTS_DIR / "error_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))

    md = ["# Error examples (test set, greedy decoding)\n"]
    for cat, df in examples.items():
        md.append(f"## {cat}\n")
        for r in df.itertuples():
            md += [f"- **EN:** {r.en}  ", f"  **Ref:** {r.am}  "]
            md += [f"  **{label}:** {getattr(r, m)}  " for m, label in MODELS.items()]
            md[-1] += "\n"
    (C.RESULTS_DIR / "error_examples.md").write_text("\n".join(md))

    all_stats = {}
    for name in MODELS:
        if name == "seq2seq":
            continue
        picked, stats = attention_study(load_split("test"), name)
        all_stats[name] = {"stats": stats, "examples": picked}
        print(name, json.dumps(stats, indent=2))
    (C.RESULTS_DIR / "attention_stats.json").write_text(json.dumps(all_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
