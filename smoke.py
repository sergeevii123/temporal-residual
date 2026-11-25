"""Smoke-run: shapes and data flow only. No training, no numbers."""
import torch
from data import make_batch, show
from model import TinyTR

torch.manual_seed(0)
ids, ans = make_batch(4, seed=0)
show(ids)
m = TinyTR()
print("params:", sum(p.numel() for p in m.parameters()))
logits = m(ids)
print("ids:", tuple(ids.shape), "logits:", tuple(logits.shape), "ans:", tuple(ans.shape))
m2 = TinyTR(use_stream=False)
print("no-stream logits:", tuple(m2(ids).shape))
print("smoke OK — stream wired; update rule + training + ablation are TODOs")
