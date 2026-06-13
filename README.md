# MiniGPT

MiniGPT is a small GPT-style decoder-only transformer language model built from scratch in PyTorch. It includes character-level tokenization, causal self-attention, multi-head attention, transformer blocks, autoregressive training, checkpointing, and text generation.

This is a portfolio project designed to show practical understanding of how GPT-like language models work internally.

## Features

- Character-level tokenizer
- Train/validation data split
- Causal masked self-attention
- Multi-head attention
- Feed-forward network
- Residual connections
- Layer normalization
- Dropout
- Autoregressive next-token training
- Temperature and top-k text generation
- Apple Silicon `mps` acceleration when available
- Loss curve export

## Project Structure

```text
mini-gpt/
  README.md
  requirements.txt
  data/
    input.txt
  src/
    config.py
    dataset.py
    generate.py
    model.py
    train.py
  outputs/
    checkpoint.pt
    loss_curve.png
    sample.txt
  notebooks/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Add Training Data

Place a plain text dataset at:

```text
data/input.txt
```

Good first datasets:

- Shakespeare text
- TinyStories
- public domain books from Project Gutenberg
- your own notes or writing

For copyrighted datasets, document the source and usage carefully.

## Train

```bash
python src/train.py
```

For a quick smoke test:

```bash
python src/train.py --data data/tiny_sample.txt --max-iters 20 --block-size 32 --batch-size 4 --eval-iters 2
```

The script automatically uses:

1. `mps` on Apple Silicon when available
2. `cuda` when available
3. CPU otherwise

## Generate Text

```bash
python src/generate.py --prompt "To be or not to be" --tokens 500
```

Generated text is written to:

```text
outputs/sample.txt
```

## How It Works

MiniGPT is trained to predict the next character in a sequence. For each input sequence, the target sequence is shifted one character to the right.

The model uses causal self-attention, which means each token can only attend to itself and earlier tokens. This prevents the model from seeing the future during training and matches how generation works at inference time.

Each transformer block contains:

- layer normalization
- multi-head causal self-attention
- residual connection
- layer normalization
- feed-forward network
- residual connection

After training, the model generates text one token at a time by sampling from the probability distribution predicted for the next character.

## Portfolio Talking Points

- Implemented a decoder-only transformer from scratch in PyTorch.
- Built causal attention masking to enforce autoregressive generation.
- Added sampling controls with temperature and top-k filtering.
- Used training and validation loss to evaluate model fit.
- Designed the project to run locally on Apple Silicon using PyTorch MPS.

## Limitations

This model is intentionally small. It learns patterns from the training text but does not have broad world knowledge, instruction-following ability, or the scale of production LLMs.

Possible improvements:

- Byte-pair encoding tokenizer
- Larger dataset
- Mixed precision training
- Learning-rate scheduling
- Weights & Biases experiment tracking
- Hugging Face model export
- Streamlit or Gradio demo UI
