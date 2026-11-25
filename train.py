"""Training scaffold. TODOs for you — run smoke.py to check the plumbing."""
import torch
from data import make_batch
from model import TinyTR


def main():
    torch.manual_seed(0)
    model = TinyTR()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    # TODO: training loop. Sample batches with make_batch(); loss is CE on
    # the two answer positions only (logits at '=' predict tens, logits at
    # tens predict ones). Train a stream model and a no-stream baseline
    # (TinyTR(use_stream=False)) on the same data and steps.
    raise NotImplementedError("TODO: training loop")
    # TODO: ablation harness. Greedy-decode fresh problems and print three
    # accuracies: (a) stream on, (b) stream off (baseline model),
    # (c) stream model with the carried state zeroed at eval — the paper's
    # ablation (GSM8K 21% -> 40%, falls back when zeroed).
    raise NotImplementedError("TODO: ablation harness")


if __name__ == "__main__":
    main()
