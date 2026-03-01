


from flax import nnx
from module.flax_old import *
from .config import DEQConfig

def _fpi(f, z0, max_iter):
    """Fixed-point iteration: z_{k+1} = f(z_k) for max_iter steps."""
    return jax.lax.fori_loop(0, max_iter, lambda _, z: f(z), z0)


class DEQModel(nnx.Module):
    def __init__(self, config: DEQConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nnx.Embed(
            config.vocab_size, config.hidden_size, 
            dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(54613)
        )
        self.rotary = RotaryEmbedding(config)
        self.deq_blocks = nnx.vmap(Transformer, in_axes=(None, 0), out_axes=0)(
            config, jnp.arange(config.n_hidden_layers)
        )
        self.lm_head = nnx.Linear(
            config.hidden_size, config.vocab_size, use_bias=False,
            dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(4654)
        )

        # Store single-layer graphdef for jax.lax.scan (bypasses nnx tracking)
        _temp = Transformer(config, 0)
        _single_gd, _ = nnx.split(_temp)
        object.__setattr__(self, '_single_gd', _single_gd)

    def __call__(self, input_ids: jax.Array, attention_mask: jax.Array | None = None):
        hidden_states = self.embed_tokens(input_ids)
        x_inject = hidden_states  # input injection for the equilibrium

        position_ids = jnp.arange(input_ids.shape[-1])
        position_embeddings = self.rotary(position_ids)

        causal_mask = jnp.tril(
            jnp.ones((input_ids.shape[-1], input_ids.shape[-1]), dtype=jnp.bool_)
        )[None, None, :, :]
        if attention_mask is not None:
            _mask = jnp.where(attention_mask, True, False)[:, None, None, :]
            causal_mask = causal_mask & _mask

        # --- DEQ solve with implicit differentiation ---
        _, state = nnx.split(self.deq_blocks)
        single_gd = self._single_gd
        max_iter = self.config.deq_max_iter

        # All traced values must be explicit args (no closures) to avoid tracer leaks
        def _f(s, z, mask, pos_emb, x_inj):
            def scan_body(z, layer_state):
                layer = nnx.merge(single_gd, layer_state)
                return layer(z, mask, pos_emb), None
            z, _ = jax.lax.scan(scan_body, z, s)
            return z + x_inj

        @jax.custom_vjp
        def solve(state, z_init, x_inj, mask, pos_emb):
            return _fpi(
                lambda z: _f(state, z, mask, pos_emb, x_inj), z_init, max_iter
            )

        def solve_fwd(state, z_init, x_inj, mask, pos_emb):
            z_star = solve(state, z_init, x_inj, mask, pos_emb)
            return z_star, (z_star, state, x_inj, mask, pos_emb)

        def solve_bwd(res, g):
            z_star, state, x_inj, mask, pos_emb = res

            # VJP of f w.r.t. z at the fixed point
            _, vjp_z = jax.vjp(
                lambda z: _f(state, z, mask, pos_emb, x_inj), z_star
            )

            # Solve (I - J_f^T) v = g  via  v_{k+1} = g + J_f^T v_k
            v_star = _fpi(lambda v: g + vjp_z(v)[0], g, max_iter)

            # VJP of f w.r.t. params (state) at the fixed point
            _, vjp_s = jax.vjp(
                lambda s: _f(s, z_star, mask, pos_emb, x_inj), state
            )
            grad_state = vjp_s(v_star)[0]

            return (
                grad_state,                                    # state
                jnp.zeros_like(z_star),                        # z_init
                v_star,                                        # x_inj
                jnp.zeros_like(mask, dtype=jnp.float32),       # mask
                jax.tree.map(jnp.zeros_like, pos_emb),         # pos_emb
            )

        solve.defvjp(solve_fwd, solve_bwd)

        z_star = solve(state, hidden_states, x_inject, causal_mask, position_embeddings)
        logits = self.lm_head(z_star)
        return logits

class UModel(nnx.Module):
    def __init__(self, config: DEQConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nnx.Embed(config.vocab_size, config.hidden_size, dtype=jnp.float32, param_dtype=config.dtype, rngs=nnx.Rngs(54613))

        self.rotary = RotaryEmbedding(config)
        
        self.block = nnx.vmap(Transformer, in_axes=(None, 0), out_axes=0)(
            config, jnp.arange(config.n_hidden_layers)
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
        def scan_blocks(hidden_states, block): 
            return block(
                hidden_states,
                causal_mask,
                position_embeddding
            )

        @nnx.scan(in_axes=(nnx.Carry, None, 0), out_axes=nnx.Carry)
        def repeat_blocks(hidden_states, block, _i):
            return scan_blocks(hidden_states, block)
        
        hidden_states = repeat_blocks(hidden_states, self.block, jnp.arange(self.config.n_hidden_layers_repeat))

        logits = self.lm_head(hidden_states)

        return logits