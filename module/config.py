

import jax
import jax.numpy as jnp

from dataclasses import dataclass

from .rope import RoPEConfig


@dataclass(frozen=True)
class Config:
    vocab_size: int = 132748
    hidden_size: int = 512
    n_hidden_layer: int = 12
    intermediated_size: int = 8192
    n_layers: int = 8
    dtype: jnp.dtype = jnp.bfloat16

    # attn
    act_fn: jax.nn = jax.nn.silu
    n_heads: int = 32
    n_kv_heads: int = 8
    attn_bias: bool = False

    eps: float = 1e-6
    rope: RoPEConfig = RoPEConfig()