"""Download → clean → deduplicate → split → train SentencePiece → encode.

Run:  python -m src.prepare_data
Writes data/processed/{train,val,test}.tsv, models/spm_{en,am}.model and
results/dataset_stats.json.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm

from . import config as C
from .text import normalize_am, normalize_en

SPLITS = ["train", "validation", "test"]


def download():
    """Fetch the three parquet files from the Hugging Face hub (skipped if present).
    Plain HTTPS via curl is used because it is the most robust option on flaky networks."""
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        out = C.RAW_DIR / f"{s}.parquet"
        if out.exists():
            continue
        url = f"https://huggingface.co/datasets/{C.HF_DATASET}/resolve/main/data/{s}-00000-of-00001.parquet"
        print(f"downloading {url}")
        subprocess.run(["curl", "-sSL", "--retry", "8", "--retry-all-errors", "-o", str(out), url], check=True)


def load_raw() -> pd.DataFrame:
    frames = []
    for s in SPLITS:
        d = pd.read_parquet(C.RAW_DIR / f"{s}.parquet")
        frames.append(pd.DataFrame({
            "en": d["translation"].map(lambda x: x["en"]),
            "am": d["translation"].map(lambda x: x["am"]),
            "orig_split": s,
        }))
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    stats["raw_pairs"] = len(df)
    stats["raw_per_split"] = df["orig_split"].value_counts().to_dict()

    # The published validation split is a verbatim copy of rows from train/test,
    # so exact duplicates are counted before anything else is changed.
    stats["exact_duplicate_pairs_raw"] = int(df.duplicated(["en", "am"]).sum())
    stats["cross_split_duplicates_raw"] = int(
        df.drop_duplicates(["en", "am", "orig_split"]).duplicated(["en", "am"]).sum())

    df = df.dropna(subset=["en", "am"]).copy()
    df["en"] = df["en"].map(normalize_en)
    df["am"] = df["am"].map(normalize_am)

    empty = (df["en"] == "") | (df["am"] == "")
    stats["removed_empty"] = int(empty.sum())
    df = df[~empty]

    # A pair must be English on one side and Ethiopic on the other.
    eth_ratio = df["am"].map(lambda s: sum("ሀ" <= ch <= "፿" for ch in s) / max(1, len(s)))
    latin_ratio = df["en"].map(lambda s: sum("a" <= ch <= "z" for ch in s) / max(1, len(s)))
    wrong_script = (eth_ratio < 0.5) | (latin_ratio < 0.5)
    stats["removed_wrong_script"] = int(wrong_script.sum())
    df = df[~wrong_script]

    identical = df["en"] == df["am"]
    stats["removed_untranslated"] = int(identical.sum())
    df = df[~identical]

    before = len(df)
    df = df.drop_duplicates(["en", "am"])
    stats["removed_duplicates_after_norm"] = before - len(df)

    # Same English sentence with several Amharic translations: keep the first so
    # that no source sentence can appear in both train and test.
    before = len(df)
    df = df.drop_duplicates(["en"])
    stats["removed_duplicate_source"] = before - len(df)

    stats["clean_pairs"] = len(df)
    return df.reset_index(drop=True)


def split(df: pd.DataFrame):
    rng = np.random.default_rng(C.SEED)
    idx = rng.permutation(len(df))
    n_val, n_test = int(len(df) * C.VAL_FRAC), int(len(df) * C.TEST_FRAC)
    test = df.iloc[idx[:n_test]]
    val = df.iloc[idx[n_test:n_test + n_val]]
    train = df.iloc[idx[n_test + n_val:]]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def train_spm(sentences, prefix: Path, vocab_size: int):
    txt = prefix.with_suffix(".txt")
    txt.write_text("\n".join(sentences), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(txt), model_prefix=str(prefix), vocab_size=vocab_size,
        model_type="unigram", character_coverage=1.0,
        pad_id=C.PAD, unk_id=C.UNK, bos_id=C.BOS, eos_id=C.EOS,
        input_sentence_size=2_000_000, shuffle_input_sentence=True,
        num_threads=8, minloglevel=2,
    )
    txt.unlink()
    return spm.SentencePieceProcessor(model_file=str(prefix) + ".model")


def length_filter(df, sp_en, sp_am, max_tokens, stats, name):
    en_len = np.array([len(x) for x in sp_en.encode(df["en"].tolist())])
    am_len = np.array([len(x) for x in sp_am.encode(df["am"].tolist())])
    ratio = np.maximum(en_len, am_len) / np.maximum(1, np.minimum(en_len, am_len))
    keep = (en_len <= max_tokens) & (am_len <= max_tokens) & (ratio <= C.MAX_LEN_RATIO)
    stats[f"{name}_removed_by_length_filter"] = int((~keep).sum())
    out = df[keep].copy()
    out["en_len"], out["am_len"] = en_len[keep], am_len[keep]
    return out.reset_index(drop=True)


def describe(df):
    en_w = df["en"].str.split().str.len()
    am_w = df["am"].str.split().str.len()
    return {
        "pairs": len(df),
        "en_words_mean": round(float(en_w.mean()), 2), "en_words_median": int(en_w.median()),
        "en_words_max": int(en_w.max()),
        "am_words_mean": round(float(am_w.mean()), 2), "am_words_median": int(am_w.median()),
        "am_words_max": int(am_w.max()),
        "en_subwords_mean": round(float(df["en_len"].mean()), 2),
        "am_subwords_mean": round(float(df["am_len"].mean()), 2),
        "en_word_types": int(len(set(w for s in df["en"] for w in s.split()))),
        "am_word_types": int(len(set(w for s in df["am"] for w in s.split()))),
    }


def main():
    for d in (C.PROC_DIR, C.MODEL_DIR, C.RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    stats = {}
    download()
    df = clean(load_raw(), stats)
    train, val, test = split(df)

    # Tokenizers are learned from the training split only.
    sp_en = train_spm(train["en"], C.MODEL_DIR / "spm_en", C.SPM_VOCAB_EN)
    sp_am = train_spm(train["am"], C.MODEL_DIR / "spm_am", C.SPM_VOCAB_AM)

    train = length_filter(train, sp_en, sp_am, C.MAX_TRAIN_TOKENS, stats, "train")
    val = length_filter(val, sp_en, sp_am, C.MAX_EVAL_TOKENS, stats, "val")
    test = length_filter(test, sp_en, sp_am, C.MAX_EVAL_TOKENS, stats, "test")

    for name, d in (("train", train), ("val", val), ("test", test)):
        d[["en", "am", "en_len", "am_len"]].to_csv(C.PROC_DIR / f"{name}.tsv", sep="\t", index=False)
        stats[name] = describe(d)

    # Unknown-token rate on the test set shows how well the subword vocabularies cover unseen text.
    for lang, sp in (("en", sp_en), ("am", sp_am)):
        ids = [i for s in sp.encode(test[lang].tolist()) for i in s]
        stats[f"test_unk_rate_{lang}"] = round(sum(i == C.UNK for i in ids) / len(ids), 6)

    (C.RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
