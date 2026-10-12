"""Inspectable neural-network primitives built from elementary PyTorch tensors.

Reference implementations prioritize clarity, not speed. Production execution
continues to use optimized torch.nn and torch.nn.functional kernels by default.
"""
import math
import torch
from torch import nn


class ScratchLinear(nn.Module):
    """y = x W^T + b, broadcasting over arbitrary leading dimensions."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("linear dimensions must be positive")
        bound = in_features ** -0.5
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight.transpose(-2, -1)
        return y if self.bias is None else y + self.bias


class ScratchEmbedding(nn.Module):
    """A lookup table with explicit tensor indexing, not an embedding library."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.weight[indices]


class ScratchLayerNorm(nn.Module):
    """Normalize each token across its features, then scale and shift."""

    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        if width < 1 or eps <= 0:
            raise ValueError("invalid normalization dimensions")
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class ScratchGELU(nn.Module):
    """Exact GELU through the Gaussian error function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2)))


def scratch_softmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Stable softmax: subtract the row maximum before exponentiation."""
    shifted = scores - scores.amax(dim=dim, keepdim=True)
    numerator = shifted.exp()
    return numerator / numerator.sum(dim=dim, keepdim=True)


def scratch_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-probability using logsumexp and index gathering."""
    if logits.ndim != 2 or targets.shape != logits.shape[:1]:
        raise ValueError("expected logits [samples, classes] and targets [samples]")
    log_probabilities = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    targets = targets.long().unsqueeze(-1)
    return -log_probabilities.gather(dim=1, index=targets).mean()
