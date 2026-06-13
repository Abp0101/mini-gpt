from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

from config import GPTConfig
from dataset import CharDataset
from model import GPTLanguageModel


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    dataset: CharDataset,
    config: GPTConfig,
    device: torch.device,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            xb, yb = dataset.get_batch(split, config.batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def plot_losses(history: list[tuple[int, float, float]], output_path: Path) -> None:
    if not history or plt is None:
        return
    steps = [item[0] for item in history]
    train_losses = [item[1] for item in history]
    val_losses = [item[2] for item in history]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_losses, label="train")
    plt.plot(steps, val_losses, label="validation")
    plt.xlabel("step")
    plt.ylabel("cross entropy loss")
    plt.title("MiniGPT training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small GPT-style language model.")
    parser.add_argument("--data", default="data/input.txt", help="Path to a plain text training file.")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    args = parser.parse_args()

    config = GPTConfig()
    if args.max_iters is not None:
        config.max_iters = args.max_iters
    if args.block_size is not None:
        config.block_size = args.block_size
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.eval_iters is not None:
        config.eval_iters = args.eval_iters
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval
    if args.n_embd is not None:
        config.n_embd = args.n_embd
    if args.n_head is not None:
        config.n_head = args.n_head
    if args.n_layer is not None:
        config.n_layer = args.n_layer
    if args.dropout is not None:
        config.dropout = args.dropout

    torch.manual_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    text_path = Path(args.data)
    if not text_path.exists():
        raise FileNotFoundError(
            f"Missing {text_path}. Add a plain text dataset there, or pass --data path/to/file.txt."
        )

    text = text_path.read_text(encoding="utf-8")
    dataset = CharDataset(text, config.block_size)
    model = GPTLanguageModel(dataset.tokenizer.vocab_size, config).to(device)
    print(f"Vocabulary size: {dataset.tokenizer.vocab_size}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[tuple[int, float, float]] = []

    for step in trange(config.max_iters):
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, dataset, config, device)
            history.append((step, losses["train"], losses["val"]))
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = dataset.get_batch("train", config.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "chars": dataset.tokenizer.chars,
        },
        checkpoint_path,
    )
    metrics_path = Path("outputs/metrics.csv")
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(["step", "train_loss", "val_loss"])
        writer.writerows(history)
    plot_losses(history, Path("outputs/loss_curve.png"))
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")
    if plt is not None:
        print("Saved loss curve to outputs/loss_curve.png")
    else:
        print("Skipped loss curve because matplotlib is not installed.")


if __name__ == "__main__":
    main()
