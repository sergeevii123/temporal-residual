"""Toy task: chained addition. The running sum is never written out,
so it must be carried in hidden state across tokens.

Format: d1+d2+...+dk = t o   (tens, ones — always two answer digits)
e.g. 3+5+2=10 -> [3, +, 5, +, 2, =, 1, 0]
"""
import random

import torch

PLUS, EQ, PAD = 10, 11, 12
VOCAB = 13  # digits 0-9, '+', '=', PAD
TOK = {**{d: str(d) for d in range(10)}, PLUS: "+", EQ: "=", PAD: "_"}


def make_problem(k=None, rng=None):
    rng = rng or random
    k = k or rng.randint(3, 5)
    ds = [rng.randint(1, 9) for _ in range(k)]
    s = sum(ds)
    toks = []
    for i, d in enumerate(ds):
        if i:
            toks.append(PLUS)
        toks.append(d)
    return toks + [EQ, s // 10, s % 10], s


def make_batch(n=128, k_min=3, k_max=5, seed=None):
    rng = random.Random(seed) if seed is not None else random
    probs = [make_problem(rng.randint(k_min, k_max), rng) for _ in range(n)]
    T = max(len(t) for t, _ in probs)
    ids = torch.full((n, T), PAD, dtype=torch.long)
    ans = torch.zeros(n, 2, dtype=torch.long)
    for i, (t, s) in enumerate(probs):
        ids[i, : len(t)] = torch.tensor(t)
        ans[i] = torch.tensor([s // 10, s % 10])
    return ids, ans


def show(ids, n=3):
    for row in ids[:n]:
        print(" ".join(TOK[int(x)] for x in row if int(x) != PAD))
