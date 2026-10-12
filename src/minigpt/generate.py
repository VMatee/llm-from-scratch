"""Generate text from a checkpoint trained by `minigpt.train`."""
import argparse

import torch

from .model import ModelConfig, TinyGPT
from .tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tiny.pt")
    parser.add_argument("--prompt", default="machine ")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    # The checkpoint is locally created by our own training script; avoid untrusted pickle payloads.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    tokenizer = CharTokenizer(checkpoint["vocabulary"])
    model = TinyGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    print(tokenizer.decode(model.generate(ids, args.tokens, args.temperature)[0].tolist()))


if __name__ == "__main__":
    main()
