from dataclasses import dataclass


@dataclass
class GPTConfig:
    batch_size: int = 32
    block_size: int = 128
    max_iters: int = 3000
    eval_interval: int = 250
    eval_iters: int = 100
    learning_rate: float = 3e-4
    warmup_iters: int = 100
    min_lr: float = 3e-5
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2
    seed: int = 1337
    checkpoint_path: str = "outputs/checkpoint.pt"
    best_checkpoint_path: str = "outputs/best.pt"
