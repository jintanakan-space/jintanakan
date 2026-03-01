

import jax
import jax.numpy as jnp

from flax import nnx

from orbax import checkpoint as ocp
from pathlib import Path

import dataclasses
import cloudpickle

def rotate_half(x: jax.Array):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concat([-x2, x1], axis=-1)

def apply_rope(query: jax.Array, key: jax.Array, cos: jax.Array, sin: jax.Array):
    # [B, S, E]
    cos = jnp.expand_dims(cos, axis=-2).astype(query.dtype)
    sin = jnp.expand_dims(sin, axis=-2).astype(query.dtype)

    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    return query, key


class LanguageModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def generate(
        self, 
        input_ids: jax.Array, 
        attention_mask: jax.Array, 
        key: jax.Array | None = None,
        max_new_tokens: int = 64, 
        temperature: float = 1.0, 
        top_p: float = 1.0, 
        top_k: float = 1.0,
        eos_token_id: int | None = None
    ):
        append_mask = jnp.ones((input_ids.shape[0], 1))
        
        # Keep track of which sequences in the batch have hit EOS
        finished = jnp.zeros((input_ids.shape[0],), dtype=jnp.bool_)

        if key is None:
            temperature = 0.0

        for _ in range(max_new_tokens):
            logits = self(input_ids, attention_mask)
            last_logit = logits[:, -1, :] # shape (B, V)

            if temperature == 0.0:
                result = jax.lax.argmax(last_logit, axis=1, index_dtype=jnp.int32)
            else:
                last_logit = last_logit / temperature
                key, subkey = jax.random.split(key)
                result = jax.random.categorical(subkey, last_logit, axis=-1)
                
            result = result.astype(jnp.int32)
            
            # If sequence is finished, force padding token if we had one, or just keep EOS
            if eos_token_id is not None:
                result = jnp.where(finished, eos_token_id, result)
                finished = finished | (result == eos_token_id)
                
            result = jnp.expand_dims(result, axis=1) # shape (B, 1)

            input_ids = jnp.concat([input_ids, result], axis=-1)
            if attention_mask is not None:
                attention_mask = jnp.concat([attention_mask, append_mask], axis=-1)
                
            if eos_token_id is not None and jnp.all(finished):
                break

        return input_ids

    def save(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)

        if not path.is_absolute():
            path = path.resolve()
        
        checkpointer = ocp.StandardCheckpointer()

        _, state = nnx.split(self)
        try: 
            arch = type(self)
            config: dataclasses = self.kwargs.get("config", None)
            if config is None:
                raise Exception(
                    f"config is None type, to resolve this send config instance with super().__init__(config=config)."
                )
            
            checkpointer.save(path, state)
            checkpointer.wait_until_finished()

            with open(path / "config.bin", "wb") as config_file, \
                open(path / "arch.bin", "wb") as arch_file:
                cloudpickle.dump(config, config_file)
                cloudpickle.dump(arch, arch_file)

            return f"save model path {path}"
        except Exception as e: 
            print(e)

    @classmethod
    def load(cls, path: str | Path):
        if isinstance(path, str):
            path = Path(path)

        if not path.is_absolute():
            path = path.resolve()

        if cls is LanguageModel:
            with open(path / "config.bin", "rb") as config_file, \
                open(path /  "arch.bin", "rb") as model_file:
                config = cloudpickle.load(config_file)
                arch = cloudpickle.load(model_file)

        else:
            arch = cls
            with open(path / "config.bin", "rb") as f:
                config = cloudpickle.load(f)

        model = nnx.eval_shape(lambda: arch(config=config))
        gdef, abs_state = nnx.split(model)

        ckpter = ocp.StandardCheckpointer()
        state = ckpter.restore(path, abs_state)
        ckpter.wait_until_finished()
        model = nnx.merge(gdef, state)

        return model

    def generate_(
        self, 
        input_ids: jax.Array, 
        attention_mask: jax.Array, 
        key: jax.Array | None = None,
        max_new_tokens: int = 64, 
        temperature: float = 1.0, 
        top_p: float = 1.0, 
        top_k: float = 1.0,
        eos_token_id: int | None = None
    ):
        B, S = input_ids.shape
        max_len = S + max_new_tokens
        
        use_cache = getattr(self, "config", None) is None or getattr(self.config, "use_cache", True)
        
        if hasattr(self, 'init_cache') and use_cache:
            self.init_cache(B, max_len)

        if key is None:
            temperature = 0.0
            
        out_ids = jnp.zeros((B, max_len), dtype=jnp.int32)
        out_ids = out_ids.at[:, :S].set(input_ids)
        out_mask = jnp.zeros((B, max_len), dtype=jnp.bool_)
        out_mask = out_mask.at[:, :S].set(attention_mask.astype(jnp.bool_))
        
        position_ids = jnp.cumsum(attention_mask, axis=-1) - 1
        position_ids = jnp.maximum(position_ids, 0)
        
        logits = self(input_ids, attention_mask, position_ids)
        last_logit = logits[:, -1, :] # shape (B, V)

        class GenerationState(nnx.Module):
            def __init__(self, key, out_ids, out_mask, last_logit, finished):
                self.key = nnx.Variable(key)
                self.out_ids = nnx.Variable(out_ids)
                self.out_mask = nnx.Variable(out_mask)
                self.last_logit = nnx.Variable(last_logit)
                self.finished = nnx.Variable(finished)

        self.gen_state = GenerationState(
            key, out_ids, out_mask, last_logit, jnp.zeros((B,), dtype=jnp.bool_)
        )

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def body(module, i):
            if temperature == 0.0:
                result = jax.lax.argmax(module.gen_state.last_logit.value, axis=1, index_dtype=jnp.int32)
            else:
                l_logit = module.gen_state.last_logit.value / temperature
                module.gen_state.key.value, subkey = jax.random.split(module.gen_state.key.value)
                result = jax.random.categorical(subkey, l_logit, axis=-1)
                
            result = result.astype(jnp.int32)
            
            if eos_token_id is not None:
                result = jnp.where(module.gen_state.finished.value, eos_token_id, result)
                module.gen_state.finished.value = module.gen_state.finished.value | (result == eos_token_id)
                
            result = jnp.expand_dims(result, axis=1) # shape (B, 1)

            # Insert next token into sequence and update mask
            module.gen_state.out_ids.value = jax.lax.dynamic_update_slice(
                module.gen_state.out_ids.value, result, (0, S + i)
            )
            append_mask = jnp.ones((B, 1), dtype=jnp.bool_)
            module.gen_state.out_mask.value = jax.lax.dynamic_update_slice(
                module.gen_state.out_mask.value, append_mask, (0, S + i)
            )
            
            # Static boolean subset slice trick
            indices = jnp.arange(max_len)
            valid_mask = indices <= (S + i)
            static_mask = module.gen_state.out_mask.value & valid_mask
            
            if use_cache:
                # Forward pass for next token (single token)
                next_position_ids = jnp.expand_dims(position_ids[:, -1] + 1 + i, axis=1)
                logits = module(result, static_mask, next_position_ids)
                module.gen_state.last_logit.value = logits[:, -1, :]
            else:
                # Forward pass for entire sequence so far
                cur_position_ids = jnp.cumsum(static_mask, axis=-1) - 1
                cur_position_ids = jnp.maximum(cur_position_ids, 0)
                logits = module(module.gen_state.out_ids.value, static_mask, cur_position_ids)
                
                # Extract logit precisely at the newly generated token's sequence index
                # index (S + i) is the token we just inserted, so we want logits from that step to predict the next
                # Wait, module(ids) returns predictions. The token we just inserted is at S+i.
                # So we want logits[:, S+i, :]
                module.gen_state.last_logit.value = jax.lax.dynamic_slice(
                    logits, (0, S + i, 0), (B, 1, logits.shape[-1])
                )[:, 0, :]
            
            return module, jnp.ones((B,))

        _, _ = body(self, jnp.arange(max_new_tokens))
        return self.gen_state.out_ids.value