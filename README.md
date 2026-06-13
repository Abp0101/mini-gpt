# MiniGPT

MiniGPT is a small GPT-style decoder-only transformer language model built from scratch in PyTorch. It includes character-level tokenization, causal self-attention, multi-head attention, transformer blocks, autoregressive training, checkpointing, and text generation.

This is a portfolio project designed to show practical understanding of how GPT-like language models work internally.

## Results

I trained a 0.82M parameter version on the Tiny Shakespeare dataset for 1,200 steps.

![Training loss curve](assets/loss_curve.png)

| Step | Train loss | Validation loss |
| ---: | ---: | ---: |
| 0 | 4.2010 | 4.1996 |
| 200 | 2.4937 | 2.4969 |
| 400 | 2.3748 | 2.3757 |
| 600 | 2.1899 | 2.2053 |
| 800 | 2.0814 | 2.1080 |
| 1000 | 2.0038 | 2.0585 |
| 1199 | 1.9262 | 2.0300 |

Example generated text is available in [docs/sample_shakespeare.txt](docs/sample_shakespeare.txt). The output is intentionally imperfect because this is a small character-level model trained locally, but it learns Shakespeare-like structure, names, spacing, and word patterns.

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
  assets/
    loss_curve.png
  data/
    shakespeare.txt
    tiny_sample.txt
  docs/
    metrics.csv
    sample_shakespeare.txt
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

## Training Data

This repo includes:

- `data/tiny_sample.txt` for quick smoke tests
- `data/shakespeare.txt` for the portfolio training run

The Shakespeare dataset is public domain text commonly used for character-level language model experiments.

You can also train on your own plain text file by passing `--data path/to/file.txt`.

For copyrighted datasets, document the source and usage carefully before publishing results.

## Train

```bash
python src/train.py --data data/shakespeare.txt
```

For a quick smoke test:

```bash
python src/train.py --data data/tiny_sample.txt --max-iters 20 --block-size 32 --batch-size 4 --eval-iters 2
```

Command used for the included training result:

```bash
python src/train.py \
  --data data/shakespeare.txt \
  --max-iters 1200 \
  --block-size 64 \
  --batch-size 32 \
  --eval-iters 25 \
  --eval-interval 200 \
  --n-embd 128 \
  --n-head 4 \
  --n-layer 4
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

## Architecture

```mermaid
flowchart LR
    A[Raw text] --> B[Character tokenizer]
    B --> C[Token IDs]
    C --> D[Token embedding + positional embedding]
    D --> E[Transformer blocks]
    E --> F[Final layer norm]
    F --> G[Linear language-model head]
    G --> H[Next-character logits]
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

## License

This project is released under the [MIT License](LICENSE).
