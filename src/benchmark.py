"""Isolated training-throughput benchmark.

The two models were trained concurrently on one GPU, so their wall-clock
training times include contention. This script times each model alone on the
same batches, giving a fair per-epoch cost.

Run:  python -m src.benchmark   → results/speed_benchmark.json
"""
import json
import math
import time

import torch
import torch.nn as nn

from . import config as C
from .data import ParallelData, device, load_split, load_tokenizers
from .evaluate import sync
from .models import MODELS

WARMUP, TIMED = 10, 150


def main():
    dev = device()
    sp_en, sp_am = load_tokenizers()
    train = load_split("train")
    data = ParallelData(train.sample(C.BATCH_SIZE * (WARMUP + TIMED), random_state=0), sp_en, sp_am)
    n_batches_epoch = math.ceil(len(train) / C.BATCH_SIZE)
    out = {"device": str(dev), "batch_size": C.BATCH_SIZE, "timed_batches": TIMED}
    for name, cls in MODELS.items():
        torch.manual_seed(0)
        model = cls(sp_en.get_piece_size(), sp_am.get_piece_size()).to(dev).train()
        opt = torch.optim.Adam(model.parameters(), lr=C.LR)
        crit = nn.CrossEntropyLoss(ignore_index=C.PAD)
        for i, (_, b) in enumerate(data.batches(C.BATCH_SIZE, shuffle=False)):
            if i == WARMUP:
                sync(dev)
                t0 = time.perf_counter()
            src, tgt = b["src"].to(dev), b["tgt"].to(dev)
            logits, _ = model(src, b["src_len"], tgt[:, :-1])
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sync(dev)
        per_batch = (time.perf_counter() - t0) / TIMED
        hist = json.loads((C.RESULTS_DIR / f"history_{name}.json").read_text())
        out[name] = {"sec_per_batch": round(per_batch, 4),
                     "isolated_min_per_epoch": round(per_batch * n_batches_epoch / 60, 2),
                     "isolated_training_time_min_estimate": round(per_batch * n_batches_epoch * hist["epochs_run"] / 60, 1)}
        print(name, out[name], flush=True)
    (C.RESULTS_DIR / "speed_benchmark.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
