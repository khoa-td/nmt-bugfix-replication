import time
import argparse, json
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import BuildDataset
from model import Seq2Seq, SOS, EOS

import sys
from configs import BEST_CONFIG

ROOT = Path(__file__).resolve().parent.parent

SIZE = sys.argv[1]                                       
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None  
CONFIG = BEST_CONFIG
BEAMS = [1, 5] if LIMIT else [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def reorder(state, b):
    if isinstance(state, tuple):
        return tuple(s[:, b] for s in state)
    return state[:, b]

def trim(seq):
    if EOS in seq:
        return seq[: seq.index(EOS) + 1]
    return seq


@torch.no_grad()
def beam_search(model, src, src_len, k, max_len):
    #The Beam Search algorithm searches for the k candidates with the highest cumulative probability

    device = src.device
    enc_out, keys, mask = model.encode(src, src_len)        

    # Replicate the Encoder output for k parallel search branches   
    enc_out, keys, mask = enc_out.repeat(k, 1, 1), keys.repeat(k, 1, 1), mask.repeat(k, 1)

    state = model.init_state(k, device)
    ctx = torch.zeros(k, 2 * model.cfg["units"], device=device)
    tokens = torch.full((k,), SOS, dtype=torch.long, device=device)

    scores = torch.zeros(k, device=device)
    scores[1:] = float("-inf")

    finished = torch.zeros(k, dtype=torch.bool, device=device)
    seqs = torch.zeros(k, 0, dtype=torch.long, device=device)

    for _ in range(max_len):
        logits, state, ctx = model.decode_step(tokens, state, ctx, enc_out, keys, mask)
        logp = F.log_softmax(logits, dim=-1)                          
        V = logp.size(1)

        logp[finished] = float("-inf") # With branch is finished: +(-inf) to keep score                           
        logp[finished, EOS] = 0.0

        cand = (scores.unsqueeze(1) + logp).view(-1)
        scores, idx = cand.topk(k)

        # Given a flat index in the range [0, k * V), derive the beam index (b) and the token ID (t).
        b = torch.div(idx, V, rounding_mode="floor")
        t = idx % V

        seqs = torch.cat([seqs[b], t.unsqueeze(1)], dim=1)
        finished = finished[b] | (t == EOS)

        tokens, ctx, state = t, ctx[b], reorder(state, b)
        if finished.all():
            break

    return [trim(s) for s in seqs.tolist()]                           
                             

DATA = ROOT / "processed" / SIZE
RUN_DIR = ROOT / "checkpoints" / f"{SIZE}_cfg10"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ck = torch.load(RUN_DIR / "best.pt", map_location=device)
model = Seq2Seq(ck["cfg"], ck["v_src"], ck["v_tgt"]).to(device)
model.load_state_dict(ck["model"])
model.eval()

ds = BuildDataset(DATA / "test" / "buggy.txt", DATA / "test" / "fixed.txt", DATA / "train" / "vocab.buggy.txt", DATA / "train" / "vocab.fixed.txt")
samples = ds.data[:LIMIT] if LIMIT else ds.data

MAX_LEN = 60 if SIZE == "small" else 110      

times_dict = {}

for k in BEAMS:
    out_file = RUN_DIR / f"predictions_k{k}.txt"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with open(out_file, "w", encoding="utf-8") as f:
        for i, (src, _) in enumerate(samples):
            src_tensor = src.unsqueeze(0).to(device)
            src_len = torch.tensor([len(src)])

            cands = beam_search(model, src_tensor, src_len, k, MAX_LEN)
            decoded_cands = [ds.fv.decode(c) for c in cands]
            f.write(" <SEP> ".join(decoded_cands) + "\n")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times_dict[k] = time.perf_counter() - start_time 
    
    print(f"Done k={k}: {times_dict[k]:.2f} seconds")


with open(RUN_DIR / "times.json", "w") as f:
    json.dump(times_dict, f)


print("Done!")
                

