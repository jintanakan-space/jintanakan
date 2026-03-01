# Deep Equilibrium (DEQ) Transformer — Summary

## Architecture

### DEQModel — Fixed-Point Equilibrium

A transformer that finds a **fixed point** `z* = f(z*) + x_inject` using implicit differentiation.

**Forward pass:**

1. Embed input → `x_inject` (input injection, anchors the equilibrium to the input)
2. Initialize `z = x_inject`
3. Repeat `deq_max_iter` times:
   - `z = scan(block_0, block_1, ..., block_N)(z) + x_inject`
4. `logits = lm_head(z*)`

**Backward pass** (O(1) memory via `jax.custom_vjp`):

1. Solve `(I - J_f^T) v = g` via fixed-point iteration: `v_{k+1} = g + J_f^T @ v_k`
2. Compute param gradients: `∂f/∂θ^T @ v_star`
3. No need to store intermediate activations — only `z*` is saved

**Key implementation details:**

- Uses `jax.lax.fori_loop` for the outer fixed-point iteration (compiled loop, not unrolled)
- Uses `jax.lax.scan` (not `nnx.scan`) for the inner block scan — required to avoid trace-level conflicts when nesting inside `fori_loop`
- Single-layer `graphdef` extracted at init via `nnx.split` on a temp `Transformer`, stored with `object.__setattr__` to bypass nnx tracking
- All traced JAX values passed explicitly through `custom_vjp` args/residuals (no closure captures) to prevent tracer leaks

### UModel — Universal Transformer (Weight-Tied)

A simpler weight-sharing variant: scan through N blocks, repeat the group M times. Standard backprop through all iterations.

- Uses `nnx.scan` for both inner (block scan) and outer (repeat) loops
- Same weights, but gradients flow through all `N × M` layer applications
- Memory scales with depth (unlike DEQ)

## Config

```python
class DEQConfig(Config):
    n_hidden_layers: int       # number of unique transformer blocks per step
    n_hidden_layers_repeat: int # (UModel only) number of times to repeat the group
    deq_max_iter: int          # (DEQModel only) fixed-point iteration count
    deq_tol: float             # convergence tolerance (unused currently)
```

## Training Results

Dataset: WikiText, tokenizer: custom SentencePiece (132K vocab), seq_len=128, batch=256

### Run 1 — Small model (hidden=512, 4 layers, 16 iters)

| Phase     | Steps     | Loss      | PPL       | Notes                                                                |
| --------- | --------- | --------- | --------- | -------------------------------------------------------------------- |
| Warmup    | 0–200     | 12.8→14.3 | 358K→1.6M | Initial instability, loss spikes                                     |
| Learning  | 200–850   | 14.3→5.92 | 1.6M→371  | Rapid descent after finding equilibria                               |
| **Spike** | **500**   | **12.2**  | **208K**  | Cascade failure — one bad iteration compounds through all 16 repeats |
| **Spike** | **900**   | **9.6**   | **14.9K** | Same pattern, quick recovery                                         |
| Plateau   | 1000–3000 | ~7.07     | ~1180     | Saturated, no further improvement                                    |

### Run 2 — Large model (hidden=2048, 8 layers, 16 iters)

Loss plateaued at approximately the same level (~7), despite 16× more parameters.

### Key Observations

1. **Loss spikes are a DEQ-specific problem**: When a gradient update pushes the first iteration off, the error compounds across all 16 repeats. Recovery is fast because the shared weights only need a small correction.

2. **More iterations ≠ more capacity**: Once the fixed point converges (typically within a few iterations), additional iterations don't add representational power. The model is limited by the capacity of its N unique layers, not depth.

3. **Increasing model size did not break the plateau**: Scaling hidden_size from 512→2048 did not significantly improve the plateau loss. This suggests the bottleneck may be:
   - The fixed-point formulation itself being too constraining
   - The input injection mechanism (simple additive) being insufficient
   - Needing significantly more training steps (only 3000 steps were tested)
   - The training hyperparameters (lr, warmup) not being optimal for the larger model

4. **Compared to standard transformers**: A loss of ~7 (ppl ~1180) on WikiText is poor. Standard transformers of similar parameter counts typically achieve loss <4 within a few thousand steps. The DEQ formulation appears to trade expressivity for memory efficiency.

## Potential Improvements to Explore

- **Anderson acceleration** instead of simple fixed-point iteration (faster convergence)
- **Damped iteration**: `z = α·f(z) + (1-α)·z` to reduce instability
- **Learned input injection** (e.g., a projection layer) instead of raw embedding addition
- **More training steps** (>10K) to see if the plateau eventually breaks
- **Lower gradient clipping** (0.5 or 0.3) to reduce spike severity
- **Compare against UModel** with identical param count to isolate DEQ overhead
