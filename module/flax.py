


from flax import nnx
import jax
import jax.numpy as jnp

from .utils import apply_rope
from .rope import ROPE_TYPE_FN
from .cache import KVCacheBase

class MLP(nnx.Module):
    def __init__(
        self, key, 
        hidden_size: int, 
        intermediate_dize: int, 
        act_fn=jax.nn.silu, 
        bias: bool=False, 
        dtype=jnp.bfloat16
    ):
        super().__init__()
        gate_key, up_key, down_key = jax.random.split(key, 3)
        self.gate_proj = nnx.Linear(
            hidden_size, intermediate_dize, 
            use_bias=bias, dtype=dtype, 
            param_dtype=dtype,
            rngs=nnx.Rngs(gate_key)
        )
        self.up_proj = nnx.Linear(
            hidden_size, intermediate_dize, 
            use_bias=bias, dtype=dtype, 
            param_dtype=dtype,
            rngs=nnx.Rngs(up_key)
        )
        self.down_proj = nnx.Linear(
            intermediate_dize, hidden_size, 
            use_bias=bias, dtype=dtype, 
            param_dtype=dtype,
            rngs=nnx.Rngs(down_key)
        )
        self.act_fn = act_fn

    def __call__(self, hidden_states: jax.Array):
        hidden_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(hidden_states)
    

class Attention(KVCacheBase):
    def __init__(
        self, key, 
        hidden_size: int, 
        attention_head: int, 
        head_dim: int, 
        kv_head: int | None = None, 
        bias: bool = False, 
        layer_idx: int | None = None,
        dtype=jnp.bfloat16,
        use_cache: bool = True
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.use_cache = use_cache

        self.attention_head = attention_head
        kv_head = kv_head if kv_head is not None else attention_head
        self.head_dim = head_dim
        self.kv_head = kv_head

        q_key, k_key, v_key, o_key = jax.random.split(key, 4)
        self.q_proj = nnx.Linear(
            hidden_size, 
            attention_head * head_dim,
            use_bias=bias, 
            dtype=dtype, 
            param_dtype=dtype, 
            rngs=nnx.Rngs(q_key)
        )
        self.k_proj = nnx.Linear(
            hidden_size, 
            kv_head * head_dim,
            use_bias=bias, 
            dtype=dtype, 
            param_dtype=dtype, 
            rngs=nnx.Rngs(k_key)
        )
        self.v_proj = nnx.Linear(
            hidden_size, 
            kv_head * head_dim,
            use_bias=bias, 
            dtype=dtype, 
            param_dtype=dtype, 
            rngs=nnx.Rngs(v_key)
        )
        self.o_proj = nnx.Linear(
            # GQA
            attention_head * head_dim \
                if attention_head >= kv_head else kv_head * head_dim,  # MQA
            hidden_size,
            use_bias=bias, 
            dtype=dtype, 
            param_dtype=dtype, 
            rngs=nnx.Rngs(o_key)
        )

    def __call__(
        self, hidden_states: jax.Array, 
        attention_mask: jax.Array | None = None, 
        position_embedding: tuple[jax.Array] | None = None
    ):
        input_shape = hidden_states.shape[:-1]
        query = self.q_proj(hidden_states).reshape(*hidden_states.shape[:-1], -1, self.head_dim)
        key = self.k_proj(hidden_states).reshape(*hidden_states.shape[:-1], -1, self.head_dim)
        value = self.v_proj(hidden_states).reshape(*hidden_states.shape[:-1], -1, self.head_dim)

        if position_embedding is not None:
            cos, sin = position_embedding
            query, key = apply_rope(query, key, cos, sin)

        if self.use_cache:
            key, value = self.update_cache(key, value)

        if attention_mask is not None and key.shape[1] > attention_mask.shape[-1]:
            pad_len = key.shape[1] - attention_mask.shape[-1]
            attention_mask = jnp.pad(attention_mask, ((0,0), (0,0), (0,0), (0, pad_len)))

        hidden_states = nnx.dot_product_attention(
            query, key, value, mask=attention_mask
        )

        hidden_states = hidden_states.reshape(*input_shape, -1)
        return self.o_proj(hidden_states)



class RotaryEmbedding(nnx.Module):
    def __init__(
        self, base: float, 
        head_dim: int, 
        max_position_embedding: int,
        rope_scaling: dict | None = None
    ):
        super().__init__()
        
        self.original_max_position_embedding = rope_scaling.get("original_max_position_embedding", max_position_embedding) if rope_scaling else max_position_embedding
        self.max_position_embedding = max_position_embedding

        init_rope_fn = self.rope_default_fn
        if rope_scaling is not None:
            rope_type = rope_scaling.get("rope_type")
            init_rope_fn = ROPE_TYPE_FN[rope_type]
        
        inv_freq, attention_factor = init_rope_fn(base, head_dim, rope_scaling)
        self.inv_freq = nnx.Cache(inv_freq)
        self.attention_factor = nnx.Cache(attention_factor)

    def rope_default_fn(self, base, head_dim, rope_scaling):
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        return inv_freq, attention_factor

    def __call__(self, hidden_states: jax.Array, position_ids: jax.Array | None = None):
        if position_ids is None:
            B, S, _ = hidden_states.shape
            position_ids = jnp.expand_dims(jnp.arange(S), axis=(0, 1)).repeat(B, axis=0).astype(float)
        else:
            B = hidden_states.shape[0]
            position_ids = jnp.expand_dims(position_ids, axis=1).astype(float) # from (B, S) -> (B, 1, S)
            
        inv_freq = jnp.expand_dims(self.inv_freq, axis=(0, -1)).repeat(B, axis=0).astype(float)

        scale = 1.0
        if self.max_position_embedding > self.original_max_position_embedding:
            scale = self.max_position_embedding / self.original_max_position_embedding

        position_ids = position_ids / scale

        freq = (inv_freq @ position_ids).transpose(0, 2, 1)
        embed = jnp.concat([freq, freq], axis=-1)
        cos = jax.lax.cos(embed)
        sin = jax.lax.sin(embed)
        return cos, sin
    

class RMSNorm(nnx.Module): 
    def __init__(self, hidden_size: int, eps: float = 1e-9):
        super().__init__()
        self.weights = nnx.Param(jnp.ones((hidden_size, ), dtype=jnp.float32))
        self.eps = eps

    def __call__(self, hidden_states: jax.Array):
        dtype = hidden_states.dtype
        hidden_states_f32 = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.pow(hidden_states_f32, 2), axis=-1, keepdims=True)
        hidden_states = hidden_states_f32 * jax.lax.rsqrt(variance + self.eps)
        return (hidden_states * self.weights).astype(dtype=dtype)