"""Train the character-level causal Transformer with held-out evaluation."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from .data import TextSplits
from .model import ModelConfig, TinyGPT
from .optim import AdamWFromScratch
from .tokenizer import CharTokenizer


@torch.no_grad()
def estimate_loss(model: TinyGPT, splits: TextSplits, split: str,
                  batch_size: int, batches: int, seed: int) -> float:
    """Evaluate a fixed random batch sequence without consuming training RNG."""
    if batches < 1:
        raise ValueError("batches must be positive")
    previous_mode = model.training
    model.eval()
    random_source = torch.Generator().manual_seed(seed)
    total = 0.0
    try:
        for _ in range(batches):
            x, y = splits.sample_batch(
                split, batch_size, model.config.block_size,
                str(next(model.parameters()).device), random_source
            )
            _, loss = model(x, y)
            total += float(loss.item())
    finally:
        model.train(previous_mode)
    return total / batches


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.eval_batches < 1:
        raise ValueError("steps, batch sizes and evaluation intervals must be positive")
    raw = Path(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(raw)
    tokens = torch.tensor(tokenizer.encode(raw), dtype=torch.long)
    splits = TextSplits.from_tokens(tokens, args.block_size, args.val_fraction)
    config = ModelConfig(tokenizer.size, args.block_size, args.n_embed, args.n_layer, args.n_head,
                         reference_ops=getattr(args, "reference_ops", False))
    model = TinyGPT(config).to(args.device)
    optimizer = (
        AdamWFromScratch(model.parameters(), lr=args.learning_rate)
        if args.optimizer == "manual"
        else torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    )
    for step in range(1, args.steps + 1):
        x, y = splits.sample_batch("train", args.batch_size, args.block_size, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            held_out = estimate_loss(model, splits, "validation", args.batch_size,
                                     args.eval_batches, args.seed + 1000)
            print(f"step={step:5d} train_cross_entropy={loss.item():.4f} "
                  f"val_cross_entropy={held_out:.4f}")
    checkpoint = {
        "model": model.cpu().state_dict(),
        "config": asdict(config),
        "vocabulary": tokenizer.chars,
        "training": {
            "seed": args.seed,
            "steps": args.steps,
            "optimizer": args.optimizer,
            "val_fraction": args.val_fraction
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Saved {output} (small character-level model; no competitive benchmark claim)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny causal Transformer from random weights")
    parser.add_argument("--data", default="data/tiny_corpus.txt")
    parser.add_argument("--output", default="checkpoints/tiny.pt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--n-embed", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--optimizer", choices=("torch", "manual"), default="torch")
    parser.add_argument("--reference-ops", action="store_true", help="inspect unfused tensor-level primitives")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
