"""Train one model.

Run:  python -m src.train --model seq2seq
      python -m src.train --model attention
Saves models/<model>.pt (best validation loss) and results/history_<model>.json.
"""
import argparse
import json
import math
import time

import torch
import torch.nn as nn

from . import config as C
from .data import ParallelData, device, load_split, load_tokenizers
from .models import MODELS, count_params


def run_epoch(model, data, criterion, dev, optimizer=None, seed=0, log_every=200):
    train = optimizer is not None
    model.train(train)
    total_loss, total_tok = 0.0, 0
    n_batches = math.ceil(len(data) / C.BATCH_SIZE)
    t0 = time.time()
    with torch.set_grad_enabled(train):
        for step, (_, b) in enumerate(data.batches(C.BATCH_SIZE, shuffle=train, seed=seed), 1):
            src, src_len, tgt = b["src"].to(dev), b["src_len"], b["tgt"].to(dev)
            logits, _ = model(src, src_len, tgt[:, :-1])
            gold = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            n_tok = int((gold != C.PAD).sum())
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), C.CLIP)
                optimizer.step()
            total_loss += loss.item() * n_tok
            total_tok += n_tok
            if train and step % log_every == 0:
                rate = step / (time.time() - t0)
                print(f"  step {step}/{n_batches}  loss {total_loss / total_tok:.3f}  "
                      f"{rate:.1f} batch/s  eta {(n_batches - step) / rate / 60:.1f} min", flush=True)
    return total_loss / total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--limit", type=int, default=0, help="use only N training pairs (smoke test)")
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    dev = device()
    sp_en, sp_am = load_tokenizers()
    train_df, val_df = load_split("train"), load_split("val")
    if args.limit:
        train_df, val_df = train_df.head(args.limit), val_df.head(args.limit // 10)
    train, val = ParallelData(train_df, sp_en, sp_am), ParallelData(val_df, sp_en, sp_am)

    model = MODELS[args.model](sp_en.get_piece_size(), sp_am.get_piece_size()).to(dev)
    n_params = count_params(model)
    print(f"{args.model}: {n_params:,} parameters on {dev}; {len(train):,} train / {len(val):,} val pairs")

    criterion = nn.CrossEntropyLoss(ignore_index=C.PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

    ckpt_path = C.MODEL_DIR / f"{args.model}.pt"
    history, best, bad_epochs = [], float("inf"), 0
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = run_epoch(model, train, criterion, dev, optimizer, seed=epoch)
        va_loss = run_epoch(model, val, criterion, dev)
        scheduler.step(va_loss)
        secs = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "train_ppl": math.exp(tr_loss), "val_ppl": math.exp(va_loss),
                        "lr": optimizer.param_groups[0]["lr"], "seconds": secs})
        print(f"epoch {epoch}: train {tr_loss:.3f} (ppl {math.exp(tr_loss):.1f})  "
              f"val {va_loss:.3f} (ppl {math.exp(va_loss):.1f})  {secs / 60:.1f} min", flush=True)
        if va_loss < best:
            best, bad_epochs = va_loss, 0
            torch.save({"model": args.model, "config": model.config, "state_dict": model.state_dict(),
                        "epoch": epoch, "val_loss": va_loss}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= C.PATIENCE:
                print("early stopping")
                break

    summary = {
        "model": args.model, "parameters": n_params, "device": str(dev),
        "train_pairs": len(train), "val_pairs": len(val),
        "training_time_sec": time.time() - train_start,
        "epochs_run": len(history), "best_epoch": min(history, key=lambda h: h["val_loss"])["epoch"],
        "best_val_loss": best,
        "hyperparameters": {
            "embedding_size": C.EMB_DIM, "hidden_units": C.HIDDEN, "layers": C.LAYERS,
            "encoder": f"bidirectional LSTM ({C.HIDDEN // 2} per direction)", "dropout": C.DROPOUT,
            "batch_size": C.BATCH_SIZE, "learning_rate": C.LR, "optimizer": "Adam",
            "lr_schedule": "ReduceLROnPlateau(factor=0.5, patience=1)",
            "max_epochs": args.epochs, "early_stopping_patience": C.PATIENCE,
            "gradient_clip": C.CLIP, "loss": "token-level cross-entropy (padding ignored)",
            "teacher_forcing_ratio": 1.0,
            "src_vocab": sp_en.get_piece_size(), "tgt_vocab": sp_am.get_piece_size(),
        },
        "history": history,
    }
    if not args.limit:
        (C.RESULTS_DIR / f"history_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
