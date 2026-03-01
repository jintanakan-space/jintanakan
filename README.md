# Jintanakan 

A place to construct any AI architecture from imagination.

## Init model

**init model from config**
```python
from models.llama import LlamaLanguageModel, LlamaConfig

import jax
import jax.numpy as jnp

config = LlamaConfig(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    attention_head=2,
    kv_head=1,
    head_dim=8
)

model = LlamaLanguageModel(config, jax.random.key(42))
```

**save model weights**
```python
model.save(path_to_save)
# save config ( config.bin )
# and model class ( arch.bin ) 
```

**load model weights**
```python
# load from original class
from models.llama import LlamaLanguageModel

model = LlamaLanguageModel.load(path_to_model)

# load from LanguageModel 
from module.utils import LanguageModel

model = LanguageModel(path_to_model)
```

## Construct language model

**model class**
```python
from module.utils import LanguageModel
from module.cache import KVCacheBase

class MyLanguageModel(KVCacheBase, LanguageModel):
    def __init__(self, ...):
        ...

    def __call__(self, ...):
        ...
```

**helper method from LanguageModel**
```python
# save/load model weights
from module.utils import LanguageModel

model = LanguageModel.load(path_to_model)
model.save(path_to_save)

# generation
inp = "Hello world"
model.generate(**tokenizer(inp), max_new_tokens=16, temperature=0.7)
```