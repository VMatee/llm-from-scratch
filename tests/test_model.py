import torch
import pytest
from minigpt import CharTokenizer, ModelConfig, TinyGPT


def small_model():
    torch.manual_seed(1)
    model = TinyGPT(ModelConfig(vocab_size=11, block_size=8, n_embed=16, n_layer=2, n_head=4))
    model.eval()
    return model


def test_tokenizer_round_trip():
    tokenizer = CharTokenizer.from_text("aba cab")
    assert tokenizer.decode(tokenizer.encode("cab")) == "cab"
    with pytest.raises(ValueError):
        tokenizer.encode("x")


def test_forward_and_train_loss():
    model = small_model()
    x = torch.randint(0, 11, (2, 6))
    logits, loss = model(x, x)
    assert logits.shape == (2, 6, 11)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_causal_attention_does_not_see_future_tokens():
    model = small_model()
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[1, 2, 3, 9, 8]])
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    assert torch.allclose(la[:, :3], lb[:, :3], atol=1e-6)


def test_generation_and_context_limit():
    model = small_model()
    output = model.generate(torch.tensor([[1, 2]]), 10)
    assert output.shape == (1, 12)
    with pytest.raises(ValueError):
        model(torch.randint(0, 11, (1, 9)))


def test_model_configuration():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=5, n_embed=15, n_head=4)
