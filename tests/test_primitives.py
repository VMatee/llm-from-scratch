"""Match educational tensor primitives to optimized PyTorch behavior."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from minigpt.model import ModelConfig, TinyGPT
from minigpt.primitives import (
    ScratchLinear, ScratchEmbedding, ScratchLayerNorm, ScratchGELU,
    scratch_softmax, scratch_cross_entropy
)


def test_manual_linear_embedding_layernorm_and_gelu():
    torch.manual_seed(21)
    data = torch.randn(3, 5, 8)
    optimized_linear = nn.Linear(8, 6)
    manual_linear = ScratchLinear(8, 6)
    manual_linear.load_state_dict(optimized_linear.state_dict())
    assert torch.allclose(manual_linear(data), optimized_linear(data), atol=1e-6)

    lookup = nn.Embedding(12, 8)
    manual_lookup = ScratchEmbedding(12, 8)
    manual_lookup.load_state_dict(lookup.state_dict())
    indices = torch.tensor([[0, 5], [7, 2]])
    assert torch.allclose(manual_lookup(indices), lookup(indices), atol=1e-6)

    norm = nn.LayerNorm(8)
    manual_norm = ScratchLayerNorm(8)
    manual_norm.load_state_dict(norm.state_dict())
    assert torch.allclose(manual_norm(data), norm(data), atol=1e-5, rtol=1e-5)

    assert torch.allclose(ScratchGELU()(data), F.gelu(data), atol=1e-6)


def test_stable_softmax_and_ce_equivalence():
    scores = torch.tensor([[100.0, 200.0, 190.0], [-1000.0, -998.0, -999.0]])
    probs = scratch_softmax(scores)
    assert torch.allclose(probs, F.softmax(scores, dim=-1), atol=1e-6)
    targets = torch.tensor([1, 2])
    assert torch.allclose(scratch_cross_entropy(scores, targets),
                          F.cross_entropy(scores, targets), atol=1e-6)


def test_complete_model_can_switch_to_explicit_primitives_without_changing_weights():
    torch.manual_seed(11)
    optimized = TinyGPT(ModelConfig(13, block_size=8, n_embed=16, n_layer=2, n_head=4))
    manual = TinyGPT(ModelConfig(13, block_size=8, n_embed=16, n_layer=2, n_head=4, reference_ops=True))
    manual.load_state_dict(optimized.state_dict())
    optimized.eval()
    manual.eval()
    x = torch.randint(0, 13, (2, 7))
    with torch.no_grad():
        baseline, loss_base = optimized(x, x)
        actual, loss_manual = manual(x, x)
    assert torch.allclose(baseline, actual, atol=3e-5, rtol=3e-5)
    assert torch.allclose(loss_base, loss_manual, atol=3e-5, rtol=3e-5)
    with torch.no_grad():
        changed = x.clone()
        changed[:, -2:] = (changed[:, -2:] + 3) % 13
        a, _ = manual(x)
        b, _ = manual(changed)
    assert torch.allclose(a[:, :-2], b[:, :-2], atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("reference_ops", [True, False])
def test_reference_and_fast_models_backpropagate(reference_ops):
    torch.manual_seed(4)
    model = TinyGPT(ModelConfig(8, block_size=4, n_embed=8, n_layer=1, n_head=2,
                                reference_ops=reference_ops))
    sample = torch.randint(0, 8, (2, 4))
    _, loss = model(sample, sample)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.lm_head.weight.grad is not None
