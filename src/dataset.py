from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CharTokenizer:
    chars: list[str]

    def __post_init__(self) -> None:
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)


class CharDataset:
    def __init__(self, text: str, block_size: int, train_split: float = 0.9) -> None:
        if len(text) < block_size + 2:
            raise ValueError("Dataset text is too short for the configured block_size.")

        self.tokenizer = CharTokenizer.from_text(text)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        split_idx = int(train_split * len(data))
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]
        self.block_size = block_size

    def get_batch(self, split: str, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)
