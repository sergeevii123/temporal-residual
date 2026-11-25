# temporal-residual (study base)

Toy-scale scaffold for The Temporal Residual (arXiv:2611.29843):
a second residual stream B, carried token to token; every sublayer reads
u = rA·A + rB·B and writes to both streams (paper eq. 1–2).

Task: chained addition — `3+5+2=10`. The running sum is never written
out, so it must be carried in hidden state across tokens.

## Run

    pip install -r requirements.txt
    python smoke.py        # shapes + data flow, no training

## TODO (yours)

- `model.py`: the carry update rule — B from each sublayer output (eq. 1).
- `train.py`: the training loop (loss on the two answer positions).
- `train.py`: the ablation harness — stream on vs off vs carried state zeroed.

CPU-only. Runs on a MacBook Pro.
