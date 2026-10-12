# Architecture note — Character GPT

```text
character tokenizer -> embeddings -> causal multihead attention -> residual MLP -> next-token softmax
```

## Fidelity and assumptions

The tiny dataset is for a smoke test only; no held-out benchmark.

Read the source implementation and the tests together. New features should
add controlled experiments or unit tests before using benchmark claims.
