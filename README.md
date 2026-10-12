# LLM from Scratch — Tiny GPT

A small **decoder-only autoregressive Transformer**, written in readable PyTorch and initialized with random weights. It implements character tokenization, positional embeddings, masked multi-head self-attention, residual blocks, next-token cross-entropy, a training loop, and stochastic generation.

> **Scope:** runnable learning implementation, **not** a frontier LLM, an instruction-tuned assistant, or evidence of competitive benchmark performance. Tiny bundled text is for a smoke test only.

## Quick start (Python 3.10+)

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m minigpt.train --steps 100 --eval-every 25 --device cpu
python -m minigpt.generate --prompt 'machine ' --tokens 80
```

Use a larger clean text corpus to see better next-character modeling; train/validation splitting is implemented; larger datasets are required for meaningful evaluation. Generation accepts only characters present in the training vocabulary.

## Architecture

`character IDs → token + position embedding → N × (LayerNorm, causal multi-head attention, residual, LayerNorm, MLP, residual) → LayerNorm → vocabulary logits`

For a prompt of length **T**, a lower-triangular mask ensures the prediction at position *t* cannot depend on tokens at positions greater than *t*. `tests/test_model.py` directly checks this property.

## Source-level implementation

The code includes a vectorized next-character sampler with a non-overlapping
train/validation split, held-out cross-entropy reporting, and an optional
AdamW optimizer implemented with explicit first- and second-moment updates.
The high-throughput PyTorch optimizer remains the default.

Optional `--reference-ops` switches linear, embedding, LayerNorm, GELU, softmax and cross entropy to transparent tensor-level implementations; default kernels remain optimized. See [the low-level primitives](docs/primitives.md).

See [the code-level implementation guide](docs/implementation.md) for
mathematics, module-by-module source references, and runnable examples.

## Reproducibility

- `--seed` controls initial randomness in PyTorch, though cross-device numerical results can differ.
- Training writes a local checkpoint under `checkpoints/` (ignored by Git).
- Tests cover tokenization, logits/loss, causality, sequence limits, and generation shape.
- `torch.load(..., weights_only=True)` is used for local checkpoints.

## Limitations and next experiments

No pretrained tokenizer, distributed training, KV cache, attention flash kernels, mixed precision, instruction alignment, safety tuning, or benchmark results. Try larger validation corpora, tokenization alternatives, and scaling-law experiments before claiming model quality.

**Reference:** Vaswani et al. (2017), *Attention Is All You Need*; Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training*.

## Interactive notebook

Open [the step-by-step notebook](notebooks/01_walkthrough.ipynb) in Jupyter after installing this repository with `python -m pip install -e ".[dev]"`.

## Architecture walk-through

See [the implementation map](docs/architecture.md) for a component-by-component overview.

## Related educational reading

Sebastian Raschka maintains a useful [independent educational reference](https://github.com/rasbt/LLMs-from-scratch). This repository uses its own small code examples and does not reproduce his book content.
