import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import CONFIGS, COMMON, BEST_CONFIG
from dataset import BuildDataset, BucketBatchSampler, pad_fn
from model import Seq2Seq, PAD


SIZE = sys.argv[1]
assert SIZE in ("small", "medium"), "Use: python train.py small|medium"

if len(sys.argv) > 2:
    MAX_STEPS = int(sys.argv[2])
else:
    MAX_STEPS = 50000 if SIZE == "small" else 55000

EVAL_EVERY = min(1000, max(1, MAX_STEPS // 2))
SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "processed" / SIZE
CKPT = ROOT / "checkpoints" / f"{SIZE}_cfg10"
CKPT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = {**COMMON, **CONFIGS[BEST_CONFIG]}


def make_loader(split, shuffle): # Make a Dataset and DataLoader for the training or evaluation set.
    ds = BuildDataset(DATA / split / "buggy.txt", DATA / split / "fixed.txt", DATA / "train" / "vocab.buggy.txt", DATA / "train" / "vocab.fixed.txt")
    sampler = BucketBatchSampler(ds, cfg["batch_size"], shuffle=shuffle)
    return ds, DataLoader(ds, batch_sampler=sampler, collate_fn=pad_fn)


def valid_loss(model, loader): 

    model.eval()
    crit_sum = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    total, count = 0.0, 0

    with torch.no_grad():
        for src, tgt, src_len, _ in loader:
            src, tgt = src.to(device), tgt.to(device)

            logits = model(src, src_len, tgt[:, :-1])
            labels = tgt[:, 1:]

            total += crit_sum(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).item()
            count += (labels != PAD).sum().item()
    model.train()
    return total / count


train_ds, train_loader = make_loader("train", True)
_, val_loader = make_loader("eval", False)

v_src, v_tgt = len(train_ds.bv.token2idx), len(train_ds.fv.token2idx)

model = Seq2Seq(cfg, v_src, v_tgt).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], eps=cfg["adam_eps"])
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

step, best_val = 0, float("inf")
last_ckpt_path = CKPT / "last.pt"

if last_ckpt_path.exists():
    ck = torch.load(last_ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["opt"])
    step = ck["step"]
    best_val = ck["best_val"]
    print("Continue from step: ", step)


model.train()
while step < MAX_STEPS:
    for src, tgt, src_len, _ in train_loader:
        src, tgt = src.to(device), tgt.to(device)

        logits = model(src, src_len, tgt[:, :-1])
        loss = criterion(logits.reshape(-1, v_tgt), tgt[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 5.0) # Use gradient clipping to avoid exploding gradient
        optimizer.step()

        step += 1
        if step % 100 == 0:
            print(f"step {step} | train loss {loss.item():.4f}")

        if step % EVAL_EVERY == 0:
            vl = valid_loss(model, val_loader)
            print(f">>> step {step} | valid loss {vl:.4f}")

            if vl < best_val:
                best_val = vl
                torch.save({"cfg": cfg, "model": model.state_dict(), "v_src": v_src, "v_tgt": v_tgt}, CKPT / "best.pt")

            torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(), "step": step, "best_val": best_val}, CKPT / "last.pt")

        if step >= MAX_STEPS:
            break

print("Done!. The best valid loss: ", best_val)