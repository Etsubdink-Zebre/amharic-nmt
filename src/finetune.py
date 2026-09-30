"""Domain fine-tuning: continue training a model (trained on habtew + MT560) on the
habtew training sentences only, at a lower learning rate, so it keeps the broader
vocabulary of the combined data but adopts the style of the habtew references.

Run:  python -m src.finetune --model bahdanau
Keeps the pre-fine-tuning checkpoint as models/<model>_pre_ft.pt, writes the fine-tuned model
to models/<model>.pt and its history to results/history_<model>_ft.json.
"""
import argparse
import json
import math
import shutil
import time

import torch
import torch.nn as nn

from . import config as C
from .data import ParallelData, device, load_split, load_tokenizers
from .prepare_data import match_key
from .train import run_epoch
from .translator import load_model

FT_LR = 3e-4
FT_EPOCHS = 2


def habtew_train():
    """The habtew part of the training split: every training pair whose English is not from MT560."""
    import pandas as pd
    from .prepare_data import detokenize_en
    from .text import normalize_en
    train = load_split("train")
    mt = pd.read_parquet(C.RAW_DIR / "mt560.parquet")["eng"].fillna("")
    mt_keys = set(mt.map(lambda s: match_key(normalize_en(detokenize_en(s)))))
    return train[~train.en.map(match_key).isin(mt_keys)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    torch.manual_seed(C.SEED)
    dev = device()
    sp_en, sp_am = load_tokenizers()
    train = ParallelData(habtew_train(), sp_en, sp_am)
    val = ParallelData(load_split("val"), sp_en, sp_am)
    ckpt, pre = C.MODEL_DIR / f"{args.model}.pt", C.MODEL_DIR / f"{args.model}_pre_ft.pt"
    shutil.copyfile(ckpt, pre)
    model = load_model(args.model, dev).train()
    print(f"fine-tuning {args.model} on {len(train):,} habtew pairs, lr {FT_LR}", flush=True)

    crit = nn.CrossEntropyLoss(ignore_index=C.PAD)
    opt = torch.optim.Adam(model.parameters(), lr=FT_LR)
    history, t_start = [], time.time()
    best = run_epoch(model, val, crit, dev)
    print(f"before: val {best:.3f} (ppl {math.exp(best):.1f})", flush=True)
    for epoch in range(1, FT_EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(model, train, crit, dev, opt, seed=100 + epoch)
        va = run_epoch(model, val, crit, dev)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "val_ppl": math.exp(va),
                        "seconds": time.time() - t0})
        print(f"ft epoch {epoch}: train {tr:.3f}  val {va:.3f} (ppl {math.exp(va):.1f})", flush=True)
        if va < best:
            best = va
            torch.save({"model": args.model, "config": model.config, "state_dict": model.state_dict(),
                        "val_loss": va, "finetuned_epochs": epoch}, ckpt)
    (C.RESULTS_DIR / f"history_{args.model}_ft.json").write_text(json.dumps(
        {"model": args.model, "habtew_pairs": len(train), "lr": FT_LR, "epochs": FT_EPOCHS,
         "time_sec": time.time() - t_start, "best_val_loss": best, "history": history}, indent=2))


if __name__ == "__main__":
    main()
