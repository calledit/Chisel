# Chisel

> *Carving signal from noise — a competitive framework for learned gradient routing in neural language models.*

---

## Overview

Training neural networks with stochastic gradient descent applies gradient updates uniformly across all parameters, regardless of whether a given parameter was causally responsible for the current batch's errors. This is a fundamental inefficiency: gradients computed from a batch of text samples contain both meaningful learning signal and noise — interference from unrelated circuits, sampling variance, and entangled parameter updates that degrade previously learned representations.

**Chisel** is(intends to be) an experimental framework that addresses this by training a small secondary network — the **Player** — to observe the residual streams of a transformer language model during a forward pass and output a coarse mask that gates gradient flow during the backward pass. The Player learns, through a competitive game, to identify which parts of the network were causally active for each batch and suppress gradient updates to uninvolved regions.

The name reflects the core intuition: rather than applying all available gradient signal indiscriminately, Chisel removes what does not belong — analogous to diffusion models denoising a signal, or a sculptor removing material to reveal the form already present.

---

## Motivation

### The Gradient Noise Problem

A gradient computed over a mini-batch conflates two distinct signals:

- **Causal signal** — the update direction that corrects the specific failure mode present in this batch
- **Noise** — gradient contributions from parameters unrelated to the current failure, batch sampling variance, and interference between circuits

Standard optimizers (Adam, RMSprop) apply hand-engineered heuristics to manage this noise — momentum, adaptive learning rates, weight decay. These are effective but operate without any knowledge of the network's internal functional organisation. They treat all parameters roughly equally, scaled by gradient history, with no awareness of which circuits were active for which inputs.

Chisel replaces these heuristics with a **learned, context-sensitive gradient filter** that routes updates to where they are needed based on direct observation of network activations.

### The Diffusion Analogy

Diffusion models learn to remove noise from a corrupted signal iteratively. The key insight is that noise has learnable statistical structure — a network can learn what noise looks like and subtract it.

Gradient noise has analogous structure. Parameters uninvolved in processing a given sample will receive gradient signal that is, in expectation, noise relative to the learning objective for that sample. A network that observes which circuits were active can learn to identify and suppress that noise — performing gradient denoising in a manner structurally similar to diffusion.

---

## Architecture

### The Board

The **Board** is the language model being trained — a standard transformer with residual stream architecture. At each layer the residual stream is updated additively by the attention block and MLP block:

```
residual_0  (token embeddings)
    ↓  attention block
residual_1  (post-attention, pre-MLP)
    ↓  MLP block
residual_2  (post-MLP)
    ↓  ...
```

The Board is the shared object that both Players act on. It is never reset between games — all training is cumulative, and each game begins from the current Board state.

### The Player

The **Player** is a small transformer network that:

1. Receives the intermediate residual streams from both observation points per layer (post-attention and post-MLP) via cross-attention
2. Outputs a compact mask vector of shape `[num_layers × 4]` scalars

Each layer contributes four scalars:

```
[attn_left, attn_right, mlp_left, mlp_right]
```

Where `left` and `right` refer to an arbitrary split of the residual dimension at the midpoint (`d_model / 2`). This split is not assumed to be semantically meaningful — it is a simple structural handle that the Player learns to use in whatever way proves useful.

For a model with `d_model = 96` and `21` layers, the Player outputs `84` scalars total. This is intentionally coarse — expressive enough to distinguish between components, small enough to be reliably learnable.

### Gradient Masking via Hooks

PyTorch backward hooks apply the Player's mask during the backward pass:

```python
def make_hook(scalar):
    def hook(grad):
        return grad * scalar
    return hook

for name, param in board.named_parameters():
    layer, block, side = resolve_param_group(name)
    scalar = player_output[layer, block, side]
    param.register_hook(make_hook(scalar))
```

The Player runs a forward pass on each batch's residual streams before `loss.backward()` is called. The mask is applied automatically as gradients flow through the Board.

### The Game

Two Players compete on a shared Board:

1. A **fixed probe set** — a held-out batch of text samples that never changes — is evaluated to establish baseline loss
2. **Player 1** produces a mask; gradient update is applied with that mask
3. The probe set is evaluated again; Player 1's score is the loss reduction achieved
4. **Player 2** produces a mask; gradient update is applied
5. Score recorded; repeat

The winner of each turn is the Player that achieved greater loss reduction on the probe set. Scores are accumulated via an **Elo rating system** that tracks relative Player quality over time.

Loss reductions are scored nonlinearly — reductions at lower absolute loss are worth more than equivalent reductions at higher loss. This incentivises Players to develop sophisticated strategies for hard refinements rather than exploiting easy early-training gains.

The Board is never reset. Games are played on a continuously improving language model, which means the task automatically becomes harder as both Players improve — providing an implicit curriculum with no manual scheduling required.

---

## Key Design Decisions

### Fixed Probe Set for Scoring

Scoring against the training batch introduces unacceptable noise — batch sampling variance can easily exceed the signal from a single mask application. The fixed probe set eliminates this confound entirely. The only variable between Player 1's score and Player 2's score is the mask applied. The ground truth is always real human text, making the evaluation criterion hard and unbreakable — not a learned approximation.

### Residual Streams, Not Full Activations

Full activation access would give the Player too much information and be prohibitively expensive to process per sample. The residual stream is the natural information channel in transformer architectures — it is what attention heads read from and write to. Observing it at two points per layer (pre- and post-MLP) gives the Player the minimum necessary information to distinguish attention block contributions from MLP contributions without exposing internal neuron activations.

### Coarse Mask Granularity

Per-parameter masking is computationally intractable and likely unlearnable at scale. The four-scalar-per-layer design gives the Player a learnable handle on the network's gross functional organisation. Finer granularity (per-head, per-neuron) may be explored in future work but is not the starting point.

### Non-Resetting Board

Resetting the Board between games wastes compute and destroys the implicit curriculum. A non-resetting Board means every game contributes to a monotonically improving language model. It also means Players must learn strategies that work across all stages of training — early-stage Players find large improvements in weak networks, late-stage Players must develop subtle strategies for already-capable networks. Elo ratings naturally stratify Players by which regime they specialise in.

---

## Potential Issues and Hurdles

### Signal-to-Noise in Player Training

The mask's effect on loss may be small relative to batch variance, even with a fixed probe set. A single gradient step is a weak intervention. The Player may struggle to attribute loss changes to its specific mask versus the accumulated effect of many prior updates. Mitigation strategies include averaging mask effects over multiple evaluation steps, curriculum starting from high learning rate regimes where individual steps have larger effects, and careful probe set curation to maximise sensitivity.

### Collapse to Uniform Masking

The Player may learn that the optimal strategy is to output a near-uniform mask — effectively doing nothing and letting standard gradient descent proceed. This is a valid local optimum if the gradient noise is sufficiently low that filtering provides no benefit. Detecting this requires monitoring mask entropy during training. Incentive structures that reward non-uniform masks may be necessary.

### Causal Confusion

The Player observes residual streams from a forward pass and produces a mask that affects the backward pass. The causal link between what the Player sees and what it should mask is indirect — the residual streams reflect what the network computed correctly and incorrectly, but the relationship between activation patterns and gradient utility is non-obvious and may require many games to learn reliably.

### Board Non-Stationarity

Because the Board is never reset, the Player is always learning on a moving target. Representations that were accurate last game may be stale this game. This is manageable early in training when the Board changes slowly, but may become problematic if learning rate schedules cause rapid weight changes. Adaptive scoring windows or Elo decay may be needed.

### Computational Overhead

Running the Player on every batch adds a forward pass per training step. The Player must be kept small enough that this overhead does not dominate training time. Cross-attention over residual streams is not free — careful implementation with caching and minimal Player depth is necessary to keep the system practical.

### Game Balance

If one Player develops a significantly stronger strategy early, it may dominate all subsequent games, reducing the competitive pressure that drives learning. Standard Elo-based matchmaking mitigates this by pairing Players of similar skill, but maintaining a sufficiently large and diverse Player population is necessary for the game to remain competitive across all training stages.

### Whether the Player Learns Anything Generalizable

The fundamental empirical question: do residual stream observations contain enough causal structure for the Player to learn a mask that generalises across samples — or does it collapse to a form of stochastic search, memorising mask patterns that happened to score well without building a transferable model of network internals? This cannot be resolved theoretically and is the primary question a minimal proof-of-concept experiment must answer.

---

## Experimental Plan

A minimal proof-of-concept requires:

- A small Board — `d_model = 96`, `21` layers, trained on a standard language modelling corpus
- A small Player — lightweight transformer with cross-attention over Board residual streams, output head projecting to `84` scalars
- A fixed probe set of approximately 512 samples, held out from training
- Two Player instances competing over a non-resetting Board
- Elo tracking and mask entropy logging throughout

The primary success criterion is whether the Player's mask correlates with known functional organisation of the Board — if high-mask regions correspond to circuits that mechanistic interpretability tools identify as active for the probe set's input types, the Player is learning something real. If the mask is uncorrelated with activation patterns, the approach requires fundamental revision.

---

## Relation to Existing Work

Chisel is conceptually adjacent to several existing research directions:

- **Learned optimizers** (Andrychowicz et al., Metz et al.) — meta-learning optimisation algorithms from data rather than hand-engineering them
- **Gradient surgery** (Yu et al.) — modifying gradient directions to reduce inter-task interference in multi-task learning
- **Mechanistic interpretability** — identifying functional circuit organisation inside transformers; Chisel attempts to learn this organisation implicitly through competitive training
- **Sparse training** — applying updates only to subsets of parameters; Chisel learns which subset dynamically per batch rather than fixing it structurally

Chisel differs from all of these in using competitive game dynamics to drive the learning of gradient routing, rather than supervised objectives, analytical gradient manipulation, or fixed structural sparsity.

---

## Status

Early concept stage. No implementation exists. The ideas described here are theoretical and have not been empirically validated. Contributions, critiques, and experimental results are welcome.

---

## License

MIT
