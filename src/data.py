"""Tokenized datasets and length-bucketed batching."""
import random

import pandas as pd
import sentencepiece as spm
import torch

from . import config as C


def load_tokenizers():
    sp_en = spm.SentencePieceProcessor(model_file=str(C.MODEL_DIR / "spm_en.model"))
    sp_am = spm.SentencePieceProcessor(model_file=str(C.MODEL_DIR / "spm_am.model"))
    return sp_en, sp_am


def load_split(name):
    return pd.read_csv(C.PROC_DIR / f"{name}.tsv", sep="\t", keep_default_na=False)


class ParallelData:
    """Holds one split as lists of token ids. Target ids are wrapped in BOS/EOS;
    source ids get EOS only."""

    def __init__(self, df, sp_en, sp_am):
        self.df = df.reset_index(drop=True)
        self.src = [ids + [C.EOS] for ids in sp_en.encode(df["en"].tolist())]
        self.tgt = [[C.BOS] + ids + [C.EOS] for ids in sp_am.encode(df["am"].tolist())]

    def __len__(self):
        return len(self.src)

    def batches(self, batch_size, shuffle, seed=0):
        """Sort by length inside large chunks so each batch has similar lengths
        (little padding) while batch order stays random."""
        idx = list(range(len(self)))
        rng = random.Random(seed)
        if shuffle:
            rng.shuffle(idx)
        chunk = batch_size * 50
        batches = []
        for i in range(0, len(idx), chunk):
            part = sorted(idx[i:i + chunk], key=lambda j: len(self.src[j]))
            batches += [part[k:k + batch_size] for k in range(0, len(part), batch_size)]
        if shuffle:
            rng.shuffle(batches)
        for b in batches:
            yield b, collate([self.src[j] for j in b], [self.tgt[j] for j in b])


def pad(seqs):
    out = torch.full((len(seqs), max(map(len, seqs))), C.PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s)
    return out


def collate(src, tgt=None):
    batch = {"src": pad(src), "src_len": torch.tensor([len(s) for s in src])}
    if tgt is not None:
        batch["tgt"] = pad(tgt)
    return batch


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
