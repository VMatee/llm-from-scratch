# How the LLM implementation works

This is an inspectable decoder-only Transformer that trains from random initialization. It does not call a pretrained model or an external Transformer framework.

## Source map

1. Character tokenizer: src/minigpt/tokenizer.py builds a deterministic vocabulary and token IDs.
2. Train/validation sampling: src/minigpt/data.py creates disjoint token spans and vectorizes sliding input/target windows.
3. Causal attention: src/minigpt/model.py explicitly computes Q, K, V, scaled QK-transpose products, causal masking, softmax, and weighted value aggregation.
4. Feed-forward/residual blocks: model.py combines attention and MLP outputs with pre-normalization and residual connections.
5. Optimizer: src/minigpt/optim.py implements AdamW moment updates using basic tensor arithmetic.
6. Training: src/minigpt/train.py reports train and held-out validation cross entropy.
7. Generation: src/minigpt/generate.py samples tokens from the saved model weights.

## The attention algorithm

For query and key matrices Q and K with per-head dimension d, the attention scores are Q @ K.T / sqrt(d). Positions to the right of the diagonal are masked so a token cannot inspect its future labels. Row-wise softmax yields weights, which are multiplied by V. Heads are concatenated and projected.

Tests verify that changing future tokens cannot change the earlier logits. This tests causality, not language quality.

## AdamW math

For each gradient g, compute m = beta1*m + (1-beta1)*g, and v = beta2*v + (1-beta2)*g^2. Bias-correct the moving averages, update weights by -lr*m_hat/(sqrt(v_hat)+eps), and separately apply weight decay.

The optional manual optimizer is meant for inspecting the algorithm. The default torch.optim.AdamW is retained for optimized hardware performance. Tests compare their numerical updates.

## Run

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m minigpt.train --steps 100 --eval-every 25
python -m minigpt.train --steps 100 --optimizer manual --output checkpoints/manual.pt
python -m minigpt.generate --prompt "machine " --tokens 80
~~~

The tiny bundled corpus is a smoke test, not a trustworthy benchmark. Validation loss is reported but no production-scale capability is claimed.

## Low-level tensor primitives

See [the explicit primitive-equations guide](primitives.md). Choose `--reference-ops` for inspectable operations; fast PyTorch kernels remain the default.
