"""Score the phase-1 checkpoints (trained on the habtew corpus only) on the current test set,
so phase 1 and phase 2 are compared on identical, unseen sentences.

Run:  python -m src.eval_phase1 archive/phase1/models   → results/phase1_scores.json
The folder must hold the phase-1 *.pt checkpoints and their spm_{en,am}.model tokenizers.
"""
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

from . import config as C
from .data import ParallelData, load_split
from .evaluate import corpus_scores
from .train import run_epoch


def main(model_dir):
    current_models = C.MODEL_DIR
    C.MODEL_DIR = Path(model_dir)            # Translator/load_tokenizers read C.MODEL_DIR at call time
    from .translator import Translator
    test = load_split("test")
    out = {"test_sentences": len(test)}
    for name in ("seq2seq", "bahdanau", "attention"):
        tr = Translator(name, torch.device("cpu"))
        loss = run_epoch(tr.model, ParallelData(test, tr.sp_en, tr.sp_am),
                         nn.CrossEntropyLoss(ignore_index=C.PAD), tr.dev)
        hyps = tr.translate_batch(test.en.tolist())
        out[name] = {**corpus_scores(hyps, test.am.tolist()), "test_loss": round(loss, 4),
                     "test_perplexity": round(math.exp(loss), 2)}
        print(name, out[name], flush=True)
    C.MODEL_DIR = current_models
    (C.RESULTS_DIR / "phase1_scores.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
