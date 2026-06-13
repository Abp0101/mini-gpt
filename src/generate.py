from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import CharTokenizer
from model import GPTLanguageModel
from train import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a trained MiniGPT checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/checkpoint.pt")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tokenizer = CharTokenizer(checkpoint["chars"])

    model = GPTLanguageModel(tokenizer.vocab_size, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = model.generate(
        idx,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(output[0])
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/sample.txt").write_text(text, encoding="utf-8")
    print(text)
    print("\nSaved sample to outputs/sample.txt")


if __name__ == "__main__":
    main()
