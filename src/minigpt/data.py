"""Character token window sampler with non-overlapping train and validation splits."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TextSplits:
    train: torch.Tensor
    validation: torch.Tensor

    @classmethod
    def from_tokens(cls, tokens: torch.Tensor, block_size: int, val_fraction: float = 0.1) -> "TextSplits":
        if tokens.ndim != 1 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a 1D torch.long tensor")
        if block_size < 1 or not 0 < val_fraction < 1:
            raise ValueError("invalid block_size or validation fraction")
        val_size = max(block_size + 1, round(len(tokens) * val_fraction))
        train_size = len(tokens) - val_size
        if train_size <= block_size:
            raise ValueError("corpus too short for disjoint training and validation windows")
        return cls(tokens[:train_size], tokens[train_size:])

    def sample_batch(self, split: str, batch_size: int, block_size: int,
                     device: str = "cpu", generator: torch.Generator | None = None):
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        if batch_size < 1 or block_size < 1:
            raise ValueError("batch_size and block_size must be positive")
        data = self.train if split == "train" else self.validation
        if len(data) <= block_size:
            raise ValueError("not enough characters for this context length")
        starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
        offsets = torch.arange(block_size)
        positions = starts[:, None] + offsets[None, :]
        return data[positions].to(device), data[positions + 1].to(device)
