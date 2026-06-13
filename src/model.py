from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import GPTConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels = x.shape

        # This is mathematically equivalent to running separate Q/K/V projections
        # for each head, applying causal attention per head, and concatenating the
        # outputs. The fused projection performs those independent head projections
        # in one batched matrix multiply, then reshapes by head.
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, time_steps, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch, time_steps, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch, time_steps, self.n_head, self.head_size).transpose(1, 2)

        weights = q @ k.transpose(-2, -1) * self.head_size**-0.5
        weights = weights.masked_fill(~self.tril[:time_steps, :time_steps], float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, time_steps, channels)
        return self.dropout(self.proj(out))


def migrate_legacy_attention_state_dict(
    state_dict: dict[str, torch.Tensor],
    config: GPTConfig,
) -> dict[str, torch.Tensor]:
    """Convert old per-head attention checkpoints to the fused qkv format."""
    migrated = dict(state_dict)
    for layer_idx in range(config.n_layer):
        prefix = f"blocks.{layer_idx}.sa"
        qkv_key = f"{prefix}.qkv.weight"
        if qkv_key in migrated:
            continue

        query_weights = []
        key_weights = []
        value_weights = []
        for head_idx in range(config.n_head):
            head_prefix = f"{prefix}.heads.{head_idx}"
            query_key = f"{head_prefix}.query.weight"
            key_key = f"{head_prefix}.key.weight"
            value_key = f"{head_prefix}.value.weight"
            if query_key not in migrated:
                break
            query_weights.append(migrated.pop(query_key))
            key_weights.append(migrated.pop(key_key))
            value_weights.append(migrated.pop(value_key))
            migrated.pop(f"{head_prefix}.tril", None)
        else:
            migrated[qkv_key] = torch.cat(
                [
                    torch.cat(query_weights, dim=0),
                    torch.cat(key_weights, dim=0),
                    torch.cat(value_weights, dim=0),
                ],
                dim=0,
            )

        migrated.pop(f"{prefix}.tril", None)

    return migrated


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(config.n_embd, config.n_head, config.block_size, config.dropout)
        self.ffwd = FeedForward(config.n_embd, config.dropout)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time_steps = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(time_steps, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch, time_steps, channels = logits.shape
            logits = logits.view(batch * time_steps, channels)
            targets = targets.view(batch * time_steps)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
