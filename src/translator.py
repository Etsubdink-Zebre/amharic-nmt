"""Inference pipeline: raw English text → normalized → SentencePiece ids →
model decoding → detokenized Amharic. Used by evaluation, analysis and the app."""
import time

import torch

from . import config as C
from .data import collate, device, load_tokenizers
from .models import MODELS, beam_decode, greedy_decode
from .text import normalize_en


def load_model(name, dev=None):
    dev = dev or device()
    ckpt = torch.load(C.MODEL_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = MODELS[ckpt["model"]](cfg["src_vocab"], cfg["tgt_vocab"], cfg["emb"],
                                  cfg["hidden"], cfg["layers"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(dev).eval()


def strip_ids(ids):
    out = []
    for i in ids:
        if i == C.EOS:
            break
        if i not in (C.PAD, C.BOS):
            out.append(i)
    return out


class Translator:
    def __init__(self, name, dev=None):
        self.dev = dev or device()
        self.name = name
        self.model = load_model(name, self.dev)
        self.sp_en, self.sp_am = load_tokenizers()

    def encode(self, sentences):
        return [ids + [C.EOS] for ids in self.sp_en.encode([normalize_en(s) for s in sentences])]

    def translate_batch(self, sentences, batch_size=128):
        """Greedy decoding, batched. Returns list of Amharic strings."""
        src_ids = self.encode(sentences)
        order = sorted(range(len(src_ids)), key=lambda i: len(src_ids[i]))
        out = [None] * len(src_ids)
        for k in range(0, len(order), batch_size):
            idx = order[k:k + batch_size]
            b = collate([src_ids[i] for i in idx])
            ids, _ = greedy_decode(self.model, b["src"].to(self.dev), b["src_len"])
            for i, row in zip(idx, ids.tolist()):
                out[i] = self.sp_am.decode(strip_ids(row))
        return out

    def translate(self, sentence, beam=5):
        """One sentence with beam search; also returns tokens and attention for display."""
        t0 = time.perf_counter()
        src = self.encode([sentence])[0]
        b = collate([src])
        ids, attn = beam_decode(self.model, b["src"].to(self.dev), b["src_len"], beam=beam)
        ids_clean = strip_ids(ids)
        result = {
            "translation": self.sp_am.decode(ids_clean),
            "model": self.name,
            "src_tokens": self.sp_en.id_to_piece(src),
            "tgt_tokens": self.sp_am.id_to_piece(ids_clean) + ["</s>"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        if attn is not None:
            result["attention"] = attn[:len(ids_clean) + 1].tolist()
        return result
