

from flax import nnx
import jax
import jax.numpy as jnp

from .config import Config


class GLUMLP(nnx.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.gate_proj = nnx.Linear(config.hidden_size, config.intermediated_size, use_bias=False, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(5466), layer_idx)))
        self.up_proj = nnx.Linear(config.hidden_size, config.intermediated_size, use_bias=False, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(4132), layer_idx)))
        self.down_proj = nnx.Linear(config.intermediated_size, config.hidden_size, use_bias=False, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(5146), layer_idx)))

        self.gate_layernorm = nnx.RMSNorm(config.intermediated_size, epsilon=config.eps, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(51460), layer_idx)))
        self.down_layernorm = nnx.RMSNorm(config.hidden_size, epsilon=config.eps, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(51461), layer_idx)))
        self.act_fn = config.act_fn


    def __call__(self, hidden_states: jax.Array):
        hidden_states = self.gate_layernorm(self.act_fn(self.gate_proj(hidden_states))) * self.up_proj(hidden_states)

        hidden_states = self.down_proj(hidden_states)
        return self.down_layernorm(hidden_states)


class Attention(nnx.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.config = config
        head_dim = config.hidden_size // config.n_heads
        self.q_proj = nnx.Linear(config.hidden_size, config.n_heads * head_dim, use_bias=config.attn_bias, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(87410), layer_idx)))
        self.k_proj = nnx.Linear(config.hidden_size, config.n_kv_heads * head_dim, use_bias=config.attn_bias, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(87411), layer_idx)))
        self.v_proj = nnx.Linear(config.hidden_size, config.n_kv_heads * head_dim, use_bias=config.attn_bias, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(87412), layer_idx)))

        self.o_proj = nnx.Linear(config.hidden_size, config.hidden_size, use_bias=config.attn_bias, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(87413), layer_idx)))

        self.q_input_layernorm = nnx.RMSNorm(head_dim, epsilon=config.eps, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(87414), layer_idx)))
        self.k_input_layernorm = nnx.RMSNorm(head_dim, epsilon=config.eps, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(87415), layer_idx)))

        self.last_layernorm = nnx.RMSNorm(config.hidden_size, epsilon=config.eps, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(87416), layer_idx)))
        
    def __call__(
        self, hidden_states: jax.Array, 
        attention_mask: jax.Array, 
        position_embeddings: tuple[jax.Array, jax.Array]
    ):
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = self.config.hidden_size // self.config.n_heads

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.reshape((batch_size, seq_len, self.config.n_heads, head_dim))
        key = key.reshape((batch_size, seq_len, self.config.n_kv_heads, head_dim))
        value = value.reshape((batch_size, seq_len, self.config.n_kv_heads, head_dim))

        query = self.q_input_layernorm(query)
        key = self.k_input_layernorm(key)

        cos, sin = position_embeddings
        
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return jnp.concatenate((-x2, x1), axis=-1)

        query = (query * cos) + (rotate_half(query) * sin)
        key = (key * cos) + (rotate_half(key) * sin)

        num_kv_groups = self.config.n_heads // self.config.n_kv_heads
        if num_kv_groups > 1:
            key = jnp.repeat(key, num_kv_groups, axis=2)
            value = jnp.repeat(value, num_kv_groups, axis=2)

        hidden_states = jax.nn.dot_product_attention(query, key, value, mask=attention_mask)
        
        hidden_states = hidden_states.reshape((batch_size, seq_len, self.config.hidden_size))

        hidden_states = self.o_proj(hidden_states)

        return self.last_layernorm(hidden_states)

class Transformer(nnx.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()

        self.attention = Attention(config, layer_idx)
        self.mlp = GLUMLP(config, layer_idx)

        self.input_layernorm = nnx.RMSNorm(config.hidden_size, epsilon=config.eps, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(51462), layer_idx)))
        self.post_attn_layernorm = nnx.RMSNorm(config.hidden_size, epsilon=config.eps, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(51463), layer_idx)))
        self.pre_mlp_layernorm = nnx.RMSNorm(config.hidden_size, epsilon=config.eps, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(51464), layer_idx)))
        self.post_mlp_layernorm = nnx.RMSNorm(config.hidden_size, epsilon=config.eps, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(51465), layer_idx)))

    def __call__(self, hidden_states: jax.Array, attention_mask: jax.Array, position_embeddings: tuple[jax.Array, jax.Array]):

        hidden_states = self.input_layernorm(hidden_states)
        residual = hidden_states

        hidden_states = self.attention(hidden_states, attention_mask, position_embeddings)
        hidden_states = self.post_attn_layernorm(hidden_states) + residual

        hidden_states = self.pre_mlp_layernorm(hidden_states)
        residual = hidden_states

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states) + residual

        return hidden_states

class GoogolRotaryEmbedding(nnx.Module):
    """Hierarchical RoPE using gear-based position decomposition."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.inv_freq = self._calc_inv_freq()

    def _calc_inv_freq(self):
        head_dim = self.config.hidden_size // self.config.n_heads
        # Base inverse frequencies (same as standard RoPE)
        inv_freq = 1.0 / (
            self.config.rope.base ** (jnp.arange(0, head_dim, 2) / head_dim)
        )
        return inv_freq

    def __call__(self, position_ids: jax.Array):
        """
        Args:
            position_ids: 1D array of absolute positions [seq_len]
        Returns:
            (cos, sin) each of shape [1, seq_len, 1, head_dim]
        """
        K = self.config.rope.K
        B = self.config.rope.B

        # Decompose each position into K hierarchical levels
        # m_k = (m // B^k) % B
        # θ_k = θ_0 · B^{-k}
        # Combined angle α = Σ m_k · θ_k

        # Build combined angles across all hierarchy levels
        # Start with zeros: [seq_len, head_dim//2]
        combined_angles = jnp.zeros(
            (position_ids.shape[0], self.inv_freq.shape[0])
        )

        for k in range(K):
            # Decompose: extract the k-th gear digit
            m_k = (position_ids // (B ** k)) % B  # [seq_len]

            # Scale frequencies for this level: θ_k = θ_0 · B^{-k}
            theta_k = self.inv_freq / (B ** k)  # [head_dim//2]

            # Accumulate: α += m_k · θ_k
            combined_angles = combined_angles + jnp.outer(m_k, theta_k)

        # Duplicate for full head_dim (cos/sin pairs)
        embed = jnp.concatenate((combined_angles, combined_angles), axis=-1)

        cos = jnp.cos(embed)[None, :, None, :].astype(jnp.float32)
        sin = jnp.sin(embed)[None, :, None, :].astype(jnp.float32)

        return cos, sin

class RotaryEmbedding(nnx.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.inv_freq = self.calc_inv_freq()

    def calc_inv_freq(self):
        head_dim = self.config.hidden_size // self.config.n_heads
        inv_freq = 1 / ( 
            self.config.rope.base ** (jnp.arange(0, head_dim, 2) / head_dim)
        )
        return inv_freq

    def __call__(self, position_ids: jax.Array):
        # Apply RoPE Scaling ratio if the target limit exceeds the original threshold
        scale = 1.0
        if self.config.rope.max_position_embeddings > self.config.rope.original_max_position_embeddings:
            scale = self.config.rope.max_position_embeddings / self.config.rope.original_max_position_embeddings
            
        position_ids = position_ids / scale
        
        inv_freq = self.inv_freq
        freqs = jnp.outer(position_ids, inv_freq)
        embed = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(embed)[None, :, None, :].astype(jnp.float32)
        sin = jnp.sin(embed)[None, :, None, :].astype(jnp.float32)
        return cos, sin

class Model(nnx.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nnx.Embed(config.vocab_size, config.hidden_size, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(54613))

        # Select RoPE implementation based on config
        if config.rope_type == "hirope":
            self.rotary = GoogolRotaryEmbedding(config)
        else:
            self.rotary = RotaryEmbedding(config)
        self.block = nnx.vmap(Transformer, in_axes=(None, 0), out_axes=0)(
            config, jnp.arange(config.n_layers)
        )
        self.lm_head = nnx.Linear(
            config.hidden_size, config.vocab_size, use_bias=False, 
            dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(4654)
        )


    def __call__(self, input_ids: jax.Array, attention_mask: jax.Array | None = None):
        # input_ids shape: [B, S]
        # attention_mask shape: [B, S]
        hidden_states = self.embed_tokens(input_ids)

        position_ids = jnp.arange(input_ids.shape[-1])
        position_embeddding = self.rotary(position_ids)

        causal_mask = jnp.tril(jnp.ones((input_ids.shape[-1], input_ids.shape[-1]), dtype=jnp.bool_))[None, None, :, :]
        if attention_mask is not None:
            _mask = jnp.where(attention_mask, True, False)[:, None, None, :]
            causal_mask = causal_mask & _mask

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def get_next_hidden_states(
            hidden_states, block
        ): 
            return block(
                hidden_states,
                causal_mask,
                position_embeddding
            )
        
        hidden_states = get_next_hidden_states(hidden_states, self.block)

        logits = self.lm_head(hidden_states)

        return logits

class ScaleddWordEmbedding(nnx.Module):
    def __init__(self):
        super().__init__()