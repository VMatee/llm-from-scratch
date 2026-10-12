"""Tiny GPT: explicit masked multi-head attention, Transformer blocks, LM head.

This demonstrates architecture, not large-scale language-model performance.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from .primitives import (ScratchLinear, ScratchEmbedding, ScratchLayerNorm, ScratchGELU,
                         scratch_softmax, scratch_cross_entropy)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 64
    n_embed: int = 64
    n_layer: int = 2
    n_head: int = 4
    dropout: float = 0.0
    reference_ops: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size < 2 or self.block_size < 1 or self.n_embed < 1:
            raise ValueError("vocab_size>=2, block_size>=1 and n_embed>=1 required")
        if self.n_head < 1 or self.n_layer < 1 or self.n_embed % self.n_head:
            raise ValueError("n_embed must be divisible by n_head; layers and heads positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.reference_ops = config.reference_ops
        linear = ScratchLinear if config.reference_ops else nn.Linear
        self.n_head = config.n_head
        self.head_dim = config.n_embed // config.n_head
        self.qkv = linear(config.n_embed, 3 * config.n_embed)
        self.proj = linear(config.n_embed, config.n_embed)
        self.drop = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, embed = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(~self.mask[:seq, :seq], torch.finfo(scores.dtype).min)
        probabilities = scratch_softmax(scores, dim=-1) if self.reference_ops else F.softmax(scores, dim=-1)
        weights = self.drop(probabilities)
        y = (weights @ v).transpose(1, 2).contiguous().view(batch, seq, embed)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        norm = ScratchLayerNorm if config.reference_ops else nn.LayerNorm
        linear = ScratchLinear if config.reference_ops else nn.Linear
        activation = ScratchGELU if config.reference_ops else nn.GELU
        self.norm1 = norm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.norm2 = norm(config.n_embed)
        self.ffn = nn.Sequential(
            linear(config.n_embed, config.n_embed * 4),
            activation(),
            linear(config.n_embed * 4, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        embedding = ScratchEmbedding if config.reference_ops else nn.Embedding
        norm = ScratchLayerNorm if config.reference_ops else nn.LayerNorm
        linear = ScratchLinear if config.reference_ops else nn.Linear
        self.token_embedding = embedding(config.vocab_size, config.n_embed)
        self.pos_embedding = embedding(config.block_size, config.n_embed)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.norm = norm(config.n_embed)
        self.lm_head = linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        length = tokens.shape[1]
        if length < 1 or length > self.config.block_size:
            raise ValueError("sequence length must be between 1 and block_size")
        position = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.pos_embedding(position)
        logits = self.lm_head(self.norm(self.blocks(x)))
        loss = None
        if targets is not None:
            if targets.shape != tokens.shape:
                raise ValueError("targets must have the same shape as tokens")
            flatten = logits.reshape(-1, logits.size(-1))
            labels = targets.reshape(-1)
            loss = (scratch_cross_entropy(flatten, labels) if self.config.reference_ops
                    else F.cross_entropy(flatten, labels))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape [batch, sequence>=1]")
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens>=0 and temperature>0 are required")
        for _ in range(max_new_tokens):
            logits, _ = self(tokens[:, -self.config.block_size:])
            scores = logits[:, -1, :] / temperature
            probs = scratch_softmax(scores, dim=-1) if self.config.reference_ops else F.softmax(scores, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
