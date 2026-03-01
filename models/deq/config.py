


from module.config import Config
from dataclasses import dataclass

@dataclass(frozen=True)
class DEQConfig(Config):
    n_hidden_layers_repeat: int = 4
    hidden_size: int = 2048
    n_hidden_layers: int = 8
    deq_max_iter: int = 16
    deq_tol: float = 1e-5