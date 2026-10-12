"""Minimal training loop for a character-level GPT model."""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from .model import ModelConfig, TinyGPT
from .tokenizer import CharTokenizer


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    raw = Path(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(raw)
    data = torch.tensor(tokenizer.encode(raw), dtype=torch.long)
    if len(data) <= args.block_size:
        raise ValueError("Training data must be longer than block_size")
    config = ModelConfig(tokenizer.size, args.block_size, args.n_embed, args.n_layer, args.n_head)
    model = TinyGPT(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for step in range(1, args.steps + 1):
        starts = torch.randint(0, len(data) - args.block_size, (args.batch_size,))
        x = torch.stack([data[i:i + args.block_size] for i in starts]).to(args.device)
        y = torch.stack([data[i + 1:i + 1 + args.block_size] for i in starts]).to(args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 5) == 0:
            print(f"step={step:4d}, training_loss={loss.item():.4f}")
    checkpoint = {"model": model.cpu().state_dict(), "config": asdict(config), "vocabulary": tokenizer.chars}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Saved {output} (toy model; no benchmark claim)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny character-level GPT from random initialization")
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
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be >= 1")
    train(args)


if __name__ == "__main__":
    main()
