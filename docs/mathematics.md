# The mathematics behind the Transformer

*An equation-by-equation explanation of the implementation in this repository.*

This project trains a compact language model from random initialization. The aim is not to replicate the scale of a modern production language model, but to expose the complete chain of computations that makes autoregressive prediction possible. Every section below connects a mathematical quantity to an actual Python module, so that a reader can move between the explanation, the equations and the implementation.

## 1. The prediction problem: one character at a time

Let the training text be a sequence of character IDs:

$$
c_0,c_1,\ldots,c_{N-1},\qquad c_t\in\{0,\ldots,V-1\},
$$

where $V$ is the number of distinct characters in the training corpus. Given the characters up to position $t$, the model predicts the next character. Formally, it learns an approximation to the conditional distribution

$$
p_\theta(c_{t+1}\mid c_0,\ldots,c_t).
$$

For example, in the word **robot**, an input window of four characters is **robo** and its target is **obot**: every character is paired with the character immediately to its right. The model does not receive a pre-trained tokenizer or pre-trained weights.

The [character tokenizer](../src/minigpt/tokenizer.py) creates the vocabulary. The [window sampler](../src/minigpt/data.py) draws input and target windows from separate, contiguous training and validation spans. The two spans do not cross their boundary, although sampled windows within either span can overlap.

## 2. From character IDs to vectors

A neural network cannot directly compute attention over character strings. Each integer token ID selects a row from a trainable embedding matrix. A separate matrix supplies information about the position of each character:

$$
E\in\mathbb{R}^{V\times d},\qquad
P\in\mathbb{R}^{T_{\max}\times d},
$$

$$
x_t=E[c_t]+P[t]\in\mathbb{R}^{d}.
$$

Here $d$ is the hidden width and $T_{\max}$ is the maximum context length. For a minibatch of $B$ sequences containing $T$ tokens each, the resulting tensor has shape $B\times T\times d$.

In [model.py](../src/minigpt/model.py), these are the token and positional embedding modules. The optional [ScratchEmbedding](../src/minigpt/primitives.py) makes the lookup operation explicit: indexing a weight matrix is sufficient; no pre-trained embedding service is involved.

## 3. Why attention needs queries, keys and values

A token benefits from context. However, it should not attend equally to every previous token. Attention provides trainable relevance weights.

For one attention head, the input vectors are projected into three spaces:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V_h=XW_V.
$$

Using $V_h$ for the **value tensor** avoids confusing it with the vocabulary size $V$. For head dimension $d_h=d/H$, where $H$ is the number of heads, the unmasked similarity scores are

$$
S_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}.
$$

The division by $\sqrt{d_h}$ controls how rapidly dot products grow as the head dimension increases. The query at position $i$ can consult the key at position $j$ only when $j\le i$. That restriction is imposed by the causal mask:

$$
M_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

The attention distribution and head output are then

$$
A_{ij}=\frac{\exp(S_{ij}+M_{ij})}
{\sum_{k=0}^{T-1}\exp(S_{ik}+M_{ik})},
\qquad
Z=AV_h.
$$

The implementation represents masked positions by the smallest finite value in the score tensor's dtype, which has the same practical purpose of assigning negligible softmax weight to inaccessible future positions.

### A numerical attention example

Suppose a token has two accessible attention scores, $[2,1]$. Their normalized weights are

$$
a_1=\frac{e^2}{e^2+e^1}\approx0.7311,
\qquad
a_2=\frac{e^1}{e^2+e^1}\approx0.2689.
$$

If the associated one-dimensional values are $[4,10]$, the attended value is approximately

$$
z=0.7311(4)+0.2689(10)=5.6136.
$$

This small example is illustrative: the actual model computes many such weighted combinations in parallel using tensors.

With $H$ heads, the separate outputs are concatenated and mapped back to the hidden width:

$$
\operatorname{MHA}(X)
=\operatorname{Concat}(Z_1,\ldots,Z_H)W_O.
$$

These operations are visible in **CausalSelfAttention.forward** in [model.py](../src/minigpt/model.py): a QKV projection, tensor reshaping, matrix multiplication, masking, softmax and the output projection. No ready-made Transformer layer is used.

## 4. The Transformer block

Attention alone is not the entire layer. Each block also contains normalization, residual connections and a feed-forward transformation. This repository uses a *pre-normalization* block:

$$
U=X+\operatorname{MHA}(\operatorname{LN}(X)),
$$

$$
Y=U+W_2\,\operatorname{GELU}(W_1\operatorname{LN}(U)+b_1)+b_2.
$$

The equation is written for one token's features, with the feed-forward network applied independently at every sequence position. The hidden feed-forward dimension is $4d$ in this implementation.

For a feature vector $x\in\mathbb{R}^{d}$, layer normalization is

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,\qquad
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2,
$$

$$
\operatorname{LN}(x)_i
=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
$$

The network uses the exact Gaussian error function form of GELU:

$$
\operatorname{GELU}(x)
=\frac{x}{2}\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right).
$$

[primitives.py](../src/minigpt/primitives.py) includes explicit linear, embedding, layer-normalization, GELU and softmax implementations. Set the command-line flag **--reference-ops** to select these inspectable tensor-level operations. The normal path continues to use optimized PyTorch kernels because a clearer implementation is not automatically a faster one.

## 5. Converting hidden states to probabilities

After the final Transformer block, the model projects each hidden state onto one logit per vocabulary character:

$$
\ell_t=W_{\mathrm{out}}\operatorname{LN}(y_t)+b_{\mathrm{out}}
\quad\text{(where the implemented output projection has no bias).}
$$

In this repository, the final linear projection is bias-free, so $b_{\mathrm{out}}=0$. Probabilities follow from softmax:

$$
p_\theta(c_{t+1}=j\mid c_{\le t})
=\frac{\exp(\ell_{t,j})}{\sum_{k=0}^{V-1}\exp(\ell_{t,k})}.
$$

Let $y_{b,t}$ denote the correct next-token ID for batch element $b$ and position $t$. The training objective is the mean negative log probability of those IDs:

$$
\mathcal{L}_{\mathrm{CE}}
=-\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}
\log p_\theta(y_{b,t}\mid c_{b,\le t}).
$$

Consider a three-character vocabulary with logits $[2,1,0]$. If the first character is the correct answer, its probability is $e^2/(e^2+e^1+e^0)\approx0.6652$, and its cross-entropy loss is about $0.4076$ nats. Lower loss means greater assigned probability to the correct next characters; it does **not** establish general intelligence or useful language generation.

The loss is computed in **TinyGPT.forward** in [model.py](../src/minigpt/model.py). The reference-mode version in [primitives.py](../src/minigpt/primitives.py) explicitly uses log-sum-exp and gathers the correct class's log probability.

## 6. Gradient descent and decoupled AdamW

The training loop calls backpropagation to obtain gradients. By default, PyTorch performs the optimized AdamW update. An optional second optimizer in [optim.py](../src/minigpt/optim.py) exposes the update equations.

For gradient $g_t=\nabla_\theta\mathcal{L}_t$:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

The moving averages begin at zero, so they are corrected for their initial bias:

$$
\widehat{m}_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat{v}_t=\frac{v_t}{1-\beta_2^t}.
$$

The parameter update uses decoupled weight decay:

$$
\theta_{t+1}
=(1-\eta\lambda)\theta_t
-\eta\frac{\widehat{m}_t}{\sqrt{\widehat{v}_t}+\epsilon}.
$$

Here $\eta$ is the learning rate and $\lambda$ the weight-decay coefficient. A regression test applies identical gradients to the manual and PyTorch implementations and checks their numerical agreement for a small dense parameter vector. The comparison is about mathematical behavior, **not** identical throughput.

## 7. Generation and what the experiment actually demonstrates

At inference time, the network takes the available context, obtains last-position logits, scales them by a temperature $\tau>0$, and samples the next character:

$$
p_j(\tau)
=\frac{\exp(\ell_j/\tau)}
{\sum_k\exp(\ell_k/\tau)},
\qquad
c_{\mathrm{new}}\sim\operatorname{Categorical}(p_1,\ldots,p_V).
$$

The sampled character is appended to the context and the procedure repeats. The context is truncated to the model's maximum block length when necessary. This is implemented in **TinyGPT.generate** and the checkpoint-loading [generate.py](../src/minigpt/generate.py).

The [training script](../src/minigpt/train.py) reports training and held-out validation cross entropy, with validation batches drawn from a separate corpus segment. Its printed losses are *observations of a small character model*, not benchmark scores from an independent language-model evaluation suite.

### Reproduce and inspect the source

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m minigpt.train --steps 100 --eval-every 25 --device cpu
python -m minigpt.train --steps 20 --reference-ops --optimizer manual --output checkpoints/reference.pt
python -m minigpt.generate --checkpoint checkpoints/reference.pt --prompt "machine " --tokens 20
~~~

The final command requires its prompt characters to exist in the training vocabulary. The code-level overview is in [implementation.md](implementation.md), and numerical checks for the reference operations are in [test_primitives.py](../tests/test_primitives.py).

**Interpretation.** This repository demonstrates how a causal Transformer is assembled and trained from tensor operations. It does not demonstrate the accuracy, scale, efficiency or alignment properties of a production language model. Larger data, separate benchmark tasks and repeated experiments would be needed to evaluate those claims.
