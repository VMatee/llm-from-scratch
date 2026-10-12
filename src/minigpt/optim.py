"""Explicit decoupled AdamW using elementary PyTorch tensor operations.

Educational reference: optimized torch.optim.AdamW remains preferable for
large tensors and high-throughput GPU training.
"""
import math
from collections.abc import Iterable
import torch


class AdamWFromScratch:
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 3e-4,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.params = list(params)
        if not self.params:
            raise ValueError("empty parameter list")
        if not (math.isfinite(lr) and lr > 0):
            raise ValueError("lr must be positive and finite")
        if len(betas) != 2 or any(not (math.isfinite(b) and 0 <= b < 1) for b in betas):
            raise ValueError("invalid momentum parameters")
        if not (math.isfinite(eps) and eps > 0 and math.isfinite(weight_decay) and weight_decay >= 0):
            raise ValueError("invalid epsilon or weight decay")
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.state = [
            {"step": 0, "m": torch.zeros_like(p), "v": torch.zeros_like(p)}
            for p in self.params
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        beta1, beta2 = self.betas
        for p, state in zip(self.params, self.state):
            grad = p.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise ValueError("sparse gradients are not supported")
            state["step"] += 1
            step = state["step"]
            m, v = state["m"], state["v"]
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            # Bias correction compensates for both moments starting at zero.
            m_correction = 1 - beta1 ** step
            v_correction = 1 - beta2 ** step
            denom = v.sqrt().div_(math.sqrt(v_correction)).add_(self.eps)
            # Weight decay is independent of the gradient / adaptive update.
            p.mul_(1 - self.lr * self.weight_decay)
            p.addcdiv_(m, denom, value=-self.lr / m_correction)
