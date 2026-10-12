"""Regression tests for training internals and mathematical parity."""
from argparse import Namespace
import pytest
import torch
from minigpt.data import TextSplits
from minigpt.model import ModelConfig, TinyGPT
from minigpt.optim import AdamWFromScratch
from minigpt.train import estimate_loss, train


def test_split_windows_do_not_leak_into_other_partition():
    splits = TextSplits.from_tokens(torch.arange(120, dtype=torch.long), 6, 0.2)
    assert splits.train[-1].item() + 1 == splits.validation[0].item()
    generator = torch.Generator().manual_seed(13)
    for partition in ("train", "validation"):
        x, y = splits.sample_batch(partition, batch_size=20, block_size=6, generator=generator)
        assert x.shape == (20, 6)
        assert torch.equal(y, x + 1)
        if partition == "train":
            assert x.max() < splits.validation.min()
        else:
            assert x.min() >= splits.validation.min()


def test_split_input_validation():
    with pytest.raises(ValueError):
        TextSplits.from_tokens(torch.arange(15), 10)
    splits = TextSplits.from_tokens(torch.arange(120), 6)
    with pytest.raises(ValueError):
        splits.sample_batch("testing", 1, 6)


def test_manual_adamw_parity_with_optimized_reference():
    a = torch.nn.Parameter(torch.tensor([0.25, -0.4, 0.9]))
    b = torch.nn.Parameter(a.detach().clone())
    manual = AdamWFromScratch([a], lr=0.001, weight_decay=0.05)
    reference = torch.optim.AdamW([b], lr=0.001, weight_decay=0.05, foreach=False)
    for raw in ([0.4, 0.2, -0.6], [0.1, -0.7, 0.2], [0.9, 0.0, -0.1]):
        a.grad = torch.tensor(raw)
        b.grad = torch.tensor(raw)
        manual.step()
        reference.step()
    assert torch.allclose(a, b, atol=1e-7, rtol=1e-6)


def test_validation_is_repeatable_and_restores_training_mode():
    torch.manual_seed(0)
    model = TinyGPT(ModelConfig(vocab_size=8, block_size=4, n_embed=8, n_layer=1, n_head=2))
    model.train()
    splits = TextSplits.from_tokens(torch.randint(0, 8, (200,)), 4)
    first = estimate_loss(model, splits, "validation", 3, 2, 12)
    second = estimate_loss(model, splits, "validation", 3, 2, 12)
    assert first == second and model.training


@pytest.mark.parametrize("choice", ["torch", "manual"])
def test_training_saves_readable_model(tmp_path, choice):
    data = tmp_path / "text.txt"
    data.write_text("character level modelling for tests. " * 12, encoding="utf-8")
    destination = tmp_path / "model.pt"
    options = Namespace(
        seed=7, data=str(data), block_size=8, n_embed=8,
        n_layer=1, n_head=2, device="cpu", optimizer=choice,
        learning_rate=0.002, steps=2, batch_size=2, eval_every=1,
        eval_batches=2, val_fraction=0.2, output=str(destination)
    )
    train(options)
    payload = torch.load(destination, weights_only=True)
    assert payload["config"]["vocab_size"] > 2
    assert payload["training"]["optimizer"] == choice
