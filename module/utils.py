import dataclasses
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx, serialization
from orbax import checkpoint as ocp


def get_act_fn(act):
    if callable(act):
        return act
    return getattr(jax.nn, act)


def get_dtype(dtype):
    if callable(dtype):
        return dtype
    return getattr(jnp, dtype)


def rotate_half(x: jax.Array):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concat([-x2, x1], axis=-1)


def apply_rope(query: jax.Array, key: jax.Array, cos: jax.Array, sin: jax.Array):
    # [B, S, E]
    cos = jnp.expand_dims(cos, axis=-2).astype(query.dtype)
    sin = jnp.expand_dims(sin, axis=-2).astype(query.dtype)

    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    return query, key


@functools.partial(
    jax.jit, static_argnames=("temperature", "top_k", "top_p", "repetition_penalty")
)
def sample_token(
    logits, input_ids, input_mask, key, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0
):
    if temperature == 0.0:
        return jax.lax.argmax(logits, axis=1, index_dtype=jnp.int32), key

    logits = logits / temperature

    # Repetition Penalty
    if repetition_penalty != 1.0:
        one_hots = jax.nn.one_hot(input_ids, logits.shape[-1])
        valid_one_hots = jnp.where(jnp.expand_dims(input_mask, -1), one_hots, 0.0)
        score_mask = valid_one_hots.any(axis=1)
        penalized_logits = jnp.where(
            logits > 0, logits / repetition_penalty, logits * repetition_penalty
        )
        logits = jnp.where(score_mask, penalized_logits, logits)

    # Top-K Sampling
    if top_k > 0:
        top_k_vals, _ = jax.lax.top_k(logits, top_k)
        min_vals = top_k_vals[:, -1:]
        logits = jnp.where(logits < min_vals, -jnp.inf, logits)

    # Nucleus (Top-P) Sampling
    if top_p < 1.0:
        sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)

        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

        mask = cumulative_probs > top_p
        mask = jnp.roll(mask, 1, axis=-1)
        mask = mask.at[:, 0].set(False)  # Always keep at least the most probable token

        inv_indices = jnp.argsort(sorted_indices, axis=-1)
        mask_in_original_order = jnp.take_along_axis(mask, inv_indices, axis=-1)

        logits = jnp.where(mask_in_original_order, -jnp.inf, logits)

    key, subkey = jax.random.split(key)
    result = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
    return result, key


@nnx.jit
def _forward_step(model, input_ids, attention_mask, position_ids):
    return model(input_ids, attention_mask, position_ids)


class LanguageModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.config = kwargs.get("config", None)

    def set_config(self, **kwargs):
        if self.config is not None:
            if hasattr(self.config, "replace"):
                self.config = self.config.replace(**kwargs)
            else:
                for k, v in kwargs.items():
                    setattr(self.config, k, v)
            self.kwargs["config"] = self.config

    def init_cache(self, batch_size: int, max_seq_len: int):
        from .cache import KVCacheBase

        for _, module in nnx.iter_modules(self):
            if isinstance(module, KVCacheBase):
                module.init_cache_state(batch_size, max_seq_len)

    def save(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)

        if not path.is_absolute():
            path = path.resolve()

        checkpointer = ocp.StandardCheckpointer()

        _, state = nnx.split(self)
        try:
            arch = type(self)
            config = getattr(self, "config", self.kwargs.get("config", None))

            if config is None:
                raise Exception(
                    f"config is None type, to resolve this send config instance with super().__init__(config=config)."
                )

            checkpointer.save(path, state)
            checkpointer.wait_until_finished()

            with open(path / "config.json", "w") as config_file:
                json.dump(serialization.to_state_dict(config), config_file, indent=2)

            print(f"save model path {path}.")
        except Exception as e:
            print(e)

    @classmethod
    def load(cls, path: str | Path, dtype=None):
        if isinstance(path, str):
            path = Path(path)

        if not path.is_absolute():
            path = path.resolve()

        import models

        if cls is LanguageModel:
            with open(path / "config.json", "r") as config_file:
                config = json.load(config_file)

            arch = config.get("architecture")
            probably_config_class = arch.split("LanguageModel")[0].strip() + "Config"
            arch = getattr(models, arch)

        else:
            arch = cls
            with open(path / "config.json", "r") as config_file:
                config = json.load(config_file)

            arch_name = config.get("architecture", None)
            if arch_name is None:
                arch_name = arch.__name__

            probably_config_class = (
                arch_name.split("LanguageModel")[0].strip() + "Config"
            )

        config_class = getattr(models, probably_config_class, None)
        if probably_config_class is None:
            from module.config import LanguageConfig

            config_class = LanguageConfig

        config = config_class(**config)

        model = nnx.eval_shape(lambda: arch(config=config))
        gdef, abs_state = nnx.split(model)

        mesh = jax.sharding.Mesh(jax.devices(), ("model",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("model"))

        def wrap_with_sharding(leaf):
            # เช็คจำนวนมิติ (Rank) ของตัวแปร
            rank = len(leaf.shape)

            if rank == 0:
                # ถ้าเป็น Scalar (เช่น bias หรือ scale บางตัว) ห้ามใส่ชื่อแกน
                # ใช้ PartitionSpec() เปล่าๆ เพื่อบอกว่า "ไม่ต้อง shard"
                spec = jax.sharding.PartitionSpec()
            else:
                # ถ้ามีตั้งแต่ 1 มิติขึ้นไป ให้แบ่งตามแกน 'model' (จาก Mesh ของคุณ)
                # หมายเหตุ: 'model' จะไปแบ่งที่มิติแรก (Index 0) ของตัวแปรนั้น
                spec = jax.sharding.PartitionSpec("model")

            # สร้าง NamedSharding ที่ถูกต้องตามเงื่อนไข
            actual_sharding = jax.sharding.NamedSharding(mesh, spec)

            return jax.ShapeDtypeStruct(
                shape=leaf.shape, dtype=leaf.dtype, sharding=actual_sharding
            )

        # 3. Map ลงใน abs_state
        abs_state = jax.tree.map(wrap_with_sharding, abs_state)

        ckpter = ocp.StandardCheckpointer()
        state = ckpter.restore(path, abs_state)
        ckpter.wait_until_finished()

        # Cast to target dtype if specified (e.g. bf16 checkpoint → float16 for T4 GPU)
        if dtype is not None:
            target_dtype = get_dtype(dtype) if isinstance(dtype, str) else dtype
            def _cast(x):
                if hasattr(x, "dtype") and x.dtype != target_dtype and jnp.issubdtype(x.dtype, jnp.floating):
                    return x.astype(target_dtype)
                return x
            state = jax.tree.map(_cast, state)

        model = nnx.merge(gdef, state)

        return model

    def generate(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        key: jax.Array | None = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
    ):
        B, S = input_ids.shape
        max_len = S + max_new_tokens

        cfg = getattr(self, "config", getattr(self, "kwargs", {}).get("config", None))
        use_cache = True if cfg is None else getattr(cfg, "use_cache", True)

        # Enforce use_cache on all submodules (like Attention) since checkpoint config
        # might have use_cache=False and model.config.replace() doesn't update instances
        from .cache import KVCacheBase

        for _, module in nnx.iter_modules(self):
            if isinstance(module, KVCacheBase) and hasattr(module, "use_cache"):
                module.use_cache = use_cache

        if attention_mask is None:
            attention_mask = jnp.ones((B, S), dtype=jnp.bool_)
        else:
            attention_mask = attention_mask.astype(jnp.bool_)

        if hasattr(self, "init_cache") and use_cache:
            self.init_cache(B, max_len)

        if key is None:
            temperature = 0.0
            key = jax.random.key(0)

        out_ids = jnp.zeros((B, max_len), dtype=jnp.int32)
        out_ids = out_ids.at[:, :S].set(input_ids.astype(jnp.int32))
        out_mask = jnp.zeros((B, max_len), dtype=jnp.bool_)
        out_mask = out_mask.at[:, :S].set(attention_mask)

        finished = jnp.zeros((B,), dtype=jnp.bool_)

        # Prefill on prompt
        prompt_position_ids = jnp.cumsum(attention_mask.astype(jnp.int32), axis=-1) - 1
        prompt_position_ids = jnp.maximum(prompt_position_ids, 0)
        logits = _forward_step(self, input_ids, attention_mask, prompt_position_ids)
        last_logit = logits[:, -1, :]  # shape (B, V)

        # Decode loop — each step dispatches a JIT-compiled forward pass to device
        for i in range(max_new_tokens):
            next_token, key = sample_token(
                last_logit, out_ids, out_mask,
                key, temperature, top_k, top_p, repetition_penalty
            )

            if eos_token_id is not None:
                next_token = jnp.where(finished, eos_token_id, next_token)
                finished = finished | (next_token == eos_token_id)

            next_token_2d = jnp.expand_dims(next_token.astype(jnp.int32), axis=1)

            # Write generated token and mark mask as valid
            out_ids = out_ids.at[:, S + i].set(next_token.astype(jnp.int32))
            out_mask = out_mask.at[:, S + i].set(True)

            # Masking for next forward pass
            indices = jnp.arange(max_len)
            valid_mask = indices <= (S + i)
            static_mask = out_mask & valid_mask

            if use_cache:
                decode_pos = jnp.expand_dims(
                    jnp.sum(static_mask.astype(jnp.int32), axis=-1) - 1, axis=-1
                )
                logits = _forward_step(self, next_token_2d, static_mask, decode_pos)
                last_logit = logits[:, -1, :]
            else:
                cur_pos = jnp.cumsum(static_mask.astype(jnp.int32), axis=-1) - 1
                cur_pos = jnp.maximum(cur_pos, 0)
                logits = _forward_step(self, out_ids, static_mask, cur_pos)
                last_logit = jax.lax.dynamic_slice(
                    logits, (0, S + i, 0), (B, 1, logits.shape[-1])
                )[:, 0, :]

        return out_ids

