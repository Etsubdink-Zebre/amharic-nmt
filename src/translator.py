"""Inference pipeline: raw English text → normalized → SentencePiece ids →
model decoding → detokenized Amharic. Used by evaluation, analysis and the app."""
import json
import time

import torch

from . import config as C
from .data import collate, device, load_tokenizers
from .models import MODELS, beam_decode, greedy_decode
from .text import WORD_RE, normalize_en

SCOPE_NOTE = (
    "Works best on complete English sentences (about 5–30 words) about everyday topics, "
    "family, work, places, society, news and religion, which is what the training data contains. "
    "It is unreliable for single words, greetings and set phrases (\"bye\", \"hello\", \"welcome to …\"), "
    "chat and slang, rare names, technical terms and very long sentences. "
    "Common words used in an unusual sense can also be mistranslated."
)


def load_word_freq():
    path = C.MODEL_DIR / "en_word_freq.json"
    return json.loads(path.read_text()) if path.exists() else {}


def scope_warnings(text, word_freq):
    """Plain-language warnings when an input is outside what the models were trained on."""
    words = WORD_RE.findall(normalize_en(text))
    warnings = []
    if len(words) < C.MIN_INPUT_WORDS:
        warnings.append(f"Very short input ({len(words)} word{'s' if len(words) != 1 else ''}). "
                        "The model was trained on full sentences, so single words and short phrases "
                        "are often mistranslated.")
    elif text.strip()[-1:] not in ".?!;:\"'”)":
        warnings.append("Tip: end the sentence with a full stop or question mark. 94% of training "
                        "sentences do, and unpunctuated fragments tend to be translated like headlines.")
    if len(words) > C.MAX_INPUT_WORDS:
        warnings.append(f"Long input ({len(words)} words). Quality drops on sentences this long; "
                        "try splitting it into shorter sentences.")
    if word_freq:
        unseen = sorted({w for w in words if w not in word_freq})
        rare = sorted({w for w in words if 0 < word_freq.get(w, 0) < C.RARE_WORD_COUNT})
        if unseen:
            warnings.append("Never seen in the training data: " + ", ".join(f'"{w}"' for w in unseen)
                            + ". The model cannot know these words and will guess.")
        if rare:
            warnings.append("Rare in the training data: "
                            + ", ".join(f'"{w}" ({word_freq[w]}×)' for w in rare)
                            + ". These are likely to be mistranslated.")
    return warnings


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
        self.word_freq = load_word_freq()

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
            "warnings": scope_warnings(sentence, self.word_freq),
        }
        if attn is not None:
            result["attention"] = attn[:len(ids_clean) + 1].tolist()
        return result
