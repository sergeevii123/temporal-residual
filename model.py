"""Tiny decoder-only transformer + temporal residual stream (paper eq. 1-2).

STUDY BASE: the stream is wired in — carried token to token, read by every
sublayer, head reads A only. The carry UPDATE rule is a TODO for you.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


class Block(nn.Module):
    """One block, two sublayers (attn, mlp). Each reads a weighted sum of the
    two streams and writes its output to both (paper eq. 1)."""

    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.heads = heads
        # read weights rA, rB and write scale sB, one set per sublayer.
        # Init rA=1, rB=0, sB=1: the stream starts as a no-op (paper §2.1).
        self.rA1 = nn.Parameter(torch.tensor(1.0))
        self.rB1 = nn.Parameter(torch.tensor(0.0))
        self.sB1 = nn.Parameter(torch.tensor(1.0))
        self.rA2 = nn.Parameter(torch.tensor(1.0))
        self.rB2 = nn.Parameter(torch.tensor(0.0))
        self.sB2 = nn.Parameter(torch.tensor(1.0))

    def _attn(self, u, kv):
        B, d = u.shape
        h, dh = self.heads, d // self.heads
        q, k, v = self.qkv(u).chunk(3, dim=-1)
        q = q.view(B, h, 1, dh)
        k = k.view(B, h, 1, dh)
        v = v.view(B, h, 1, dh)
        ks, vs = kv
        ks.append(k)
        vs.append(v)
        y = F.scaled_dot_product_attention(q, torch.cat(ks, 2), torch.cat(vs, 2))
        return y.reshape(B, d)

    def forward(self, A, B, kv, use_stream=True):
        # attention sublayer: u = rA*A + rB*B
        u = self.rA1 * A + self.rB1 * B if use_stream else A
        y = self.o(self._attn(self.n1(u), kv))
        A = A + y
        # TODO: carry update rule (paper eq. 1) — update B from y, using sB1.
        # mlp sublayer
        u = self.rA2 * A + self.rB2 * B if use_stream else A
        y = self.mlp(self.n2(u))
        A = A + y
        # TODO: carry update rule (paper eq. 1) — update B from y, using sB2.
        return A, B, kv


class TinyTR(nn.Module):
    """Sequential recurrence (paper §2.3): one step per position, KV cache
    plus one carried vector B per sequence. Head reads A only."""

    def __init__(self, vocab=13, d=64, layers=2, heads=2, max_len=16, use_stream=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.phi = RMSNorm(d)  # carry normaliser, paper eq. (2)
        self.d, self.use_stream = d, use_stream

    def forward(self, ids):
        Bsz, T = ids.shape
        B = torch.zeros(Bsz, self.d)  # B_0(1) = 0
        kvs = [([], []) for _ in self.blocks]
        logits = []
        for t in range(T):
            A = self.emb(ids[:, t]) + self.pos.weight[t]
            for i, blk in enumerate(self.blocks):
                A, B, kvs[i] = blk(A, B, kvs[i], self.use_stream)
            logits.append(self.head(self.norm(A)))
            B = self.phi(B)  # carry to next token, paper eq. (2)
        return torch.stack(logits, dim=1)
