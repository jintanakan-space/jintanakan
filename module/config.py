from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct

from .rope import RoPEConfig


@struct.dataclass
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


@struct.dataclass
class LanguageConfig:
    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    act_fn: str
    num_hidden_layers: int
    norm_eps: float
    bias: bool
    dtype: str
    use_cache: bool

    # attn
    attention_head: int
    kv_head: int
    head_dim: int
    attention_bias: bool

    # rope
    base: float
    original_max_position_embedding: int
    max_position_embedding: int
    rope_scaling: dict | None
