
import jax
import jax.numpy as jnp
import math

from dataclasses import dataclass


# def init_rope_llama3(base, head_dim, rope_scaling: dict): 

#     inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))

#     factor = rope_scaling.get("factor", 8.0)
#     low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
#     high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
#     old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

#     low_freq_wavelen = old_context_len / low_freq_factor
#     high_freq_wavelen = old_context_len / high_freq_factor
#     wavelen = 2 * jnp.pi / inv_freq

#     inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
#     smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
#     smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
#     is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    
#     inv_freq = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
#     return inv_freq


def init_rope_llama3(base, head_dim, rope_scaling: dict):
    partial_rotary_factor = rope_scaling.get("partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / base ** (jnp.arange(0, dim, 2) / dim)

    factor = rope_scaling["factor"]  # `8` in the original implementation
    low_freq_factor = rope_scaling["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = rope_scaling["high_freq_factor"]  # `4` in the original implementation
    old_context_len = rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor

def init_rope_linear(): ...

def init_rope_dynamic(): ...

def init_rope_yarn(): ...

def init_rope_longrope(): ...

def init_rope_hirope(): ...

ROPE_TYPE_FN = {
    "linear": ...,
    "dynamic": ...,
    "yarn": ...,
    "longrope": ...,
    "llama3": init_rope_llama3,
    "hirope": ...
}

@dataclass(frozen=True)
class RoPEConfig:
    base: float = 10000.0
    dim: int = 256
    head_dim: int = 64
    max_position_embeddings: int = 8192
    original_max_position_embeddings: int = 8192
    # Hierachical RoPE params
    K: int = 3              # number of hierarchy levels (gears)
    B: int = 32             # base for position decomposition (gear capacity)
    rope_type: str = "standard"  # "standard" or "hirope"