# Explicit tensor-level primitives

Default execution uses efficient PyTorch tensor kernels. Run --reference-ops to replace linear layers, embeddings, LayerNorm, GELU, softmax and cross entropy with inspectable source-level equivalents built from basic tensor operations. All layers preserve parameter names and shapes so an existing model state dictionary can be loaded into either configuration.

| Operation | Implemented equation / technique | File |
| --- | --- | --- |
| Linear | y = x W-transposed + bias | src/minigpt/primitives.py |
| Embedding | learnable weight matrix indexed by token IDs | src/minigpt/primitives.py |
| LayerNorm | center by per-token mean, normalize with rsqrt(variance + eps), then scale/shift | src/minigpt/primitives.py |
| GELU | 0.5*x*(1+erf(x/sqrt(2))) | src/minigpt/primitives.py |
| Softmax | subtract row maximum, then normalize exponentials | src/minigpt/primitives.py |
| Cross entropy | negative gathered log-probability after logsumexp | src/minigpt/primitives.py |
| Causal attention | Q K-transposed / sqrt(head dimension), lower-triangular mask, softmax weights times V | src/minigpt/model.py |

~~~bash
python -m minigpt.train --steps 20 --reference-ops --optimizer manual --output checkpoints/reference.pt
python -m minigpt.generate --checkpoint checkpoints/reference.pt --prompt 'machine ' --tokens 20
python -m pytest -q
~~~

The optional reference kernels may be slower than fused/framework kernels on CPU or GPU. They exist to make the computations inspectable, not to claim identical latency. Tests check that both paths have numerically close logits and loss on identical parameters.
