from __future__ import annotations

import csv
import sys
from pathlib import Path

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset import CharTokenizer
from model import GPTLanguageModel, migrate_legacy_attention_state_dict
from train import get_device


def available_checkpoints() -> list[Path]:
    output_dir = ROOT / "outputs"
    candidates = [output_dir / "best.pt", output_dir / "checkpoint.pt"]
    return [path for path in candidates if path.exists()]


def latest_metrics() -> dict[str, str] | None:
    metrics_path = ROOT / "outputs" / "metrics.csv"
    if not metrics_path.exists():
        metrics_path = ROOT / "docs" / "metrics.csv"
    if not metrics_path.exists():
        return None

    with metrics_path.open(encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    return rows[-1] if rows else None


@st.cache_resource
def load_model(checkpoint_path: str) -> tuple[GPTLanguageModel, CharTokenizer, torch.device]:
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tokenizer = CharTokenizer(checkpoint["chars"])

    model = GPTLanguageModel(tokenizer.vocab_size, config).to(device)
    state_dict = migrate_legacy_attention_state_dict(checkpoint["model_state_dict"], config)
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer, device


def generate_text(
    model: GPTLanguageModel,
    tokenizer: CharTokenizer,
    device: torch.device,
    prompt: str,
    tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens=tokens,
            temperature=temperature,
            top_k=top_k,
        )
    return tokenizer.decode(output[0])


def main() -> None:
    st.set_page_config(page_title="MiniGPT", page_icon="M", layout="wide")

    st.title("MiniGPT")

    checkpoints = available_checkpoints()
    if not checkpoints:
        st.error("No checkpoint found in outputs/. Train the model first, then reload this page.")
        st.code("python src/train.py --data data/shakespeare.txt", language="bash")
        st.stop()

    metric = latest_metrics()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Checkpoint", checkpoints[0].name)
    if metric is not None:
        metric_cols[1].metric("Step", metric["step"])
        metric_cols[2].metric("Validation loss", f"{float(metric['val_loss']):.4f}")
        metric_cols[3].metric("Train loss", f"{float(metric['train_loss']):.4f}")
    else:
        metric_cols[1].metric("Step", "-")
        metric_cols[2].metric("Validation loss", "-")
        metric_cols[3].metric("Train loss", "-")

    with st.sidebar:
        checkpoint = st.selectbox("Checkpoint", checkpoints, format_func=lambda path: path.name)
        tokens = st.slider("Tokens", min_value=50, max_value=1500, value=500, step=50)
        temperature = st.slider("Temperature", min_value=0.2, max_value=1.5, value=0.8, step=0.1)
        top_k = st.slider("Top-k", min_value=1, max_value=65, value=40, step=1)

    model, tokenizer, device = load_model(str(checkpoint))

    prompt = st.text_area("Prompt", value="ROMEO:", height=140)
    invalid_chars = sorted({char for char in prompt if char not in tokenizer.stoi})
    generate = st.button("Generate", type="primary", use_container_width=True)

    if invalid_chars:
        display_chars = " ".join(repr(char) for char in invalid_chars[:8])
        st.warning(f"Prompt contains characters not in the checkpoint vocabulary: {display_chars}")

    if generate:
        if not prompt:
            st.warning("Enter a prompt first.")
        elif invalid_chars:
            st.stop()
        else:
            with st.spinner("Generating..."):
                text = generate_text(model, tokenizer, device, prompt, tokens, temperature, top_k)
            st.text_area("Output", value=text, height=520)
    else:
        sample_path = ROOT / "docs" / "sample_shakespeare.txt"
        if sample_path.exists():
            st.text_area("Output", value=sample_path.read_text(encoding="utf-8"), height=520)


if __name__ == "__main__":
    main()
