# Multi-Task Transformer: Character-Level Generation & Sentiment Analysis

This repository contains a high-performance implementation of the **Vanilla Transformer architecture** (Vaswani et al.) designed for two distinct NLP tasks: character-level language modeling and sequence classification.

## Technical Highlights & Contributions

* **Custom Transformer Implementation**: Developed a from-scratch PyTorch implementation of the Transformer architecture, including **Multi-Head Self-Attention**, **Positional Encodings**, and **Residual Connections**.
* **Character-Level Language Modeling**: Engineered a generative model trained on the **enwik8** dataset. Implemented a custom sampling engine with **Temperature Scaling** to balance text coherence and creativity.
* **Scalable Classification Pipeline**: Built a sentiment analysis classifier for the **IMDb** dataset, optimized to handle high-volume batches (up to  samples) for efficient GPU utilization.
* **Deep Observability**: Integrated **Weights & Biases (W&B)** for real-time monitoring of loss curves and **Gradient Flow Visualization** to diagnose and prevent vanishing/exploding gradients during training.
* **Modular Software Design**: Structured the codebase into a professional ML project layout (`src/models`, `src/training`, `scripts/`) to ensure high code reusability and maintainability.

## Key Features

### 1. Text Generation (enwik8)

* **Granular Prediction**: Operates at the character level to learn complex syntax and structure without a predefined vocabulary.
* **Real-time Inference**: Supports live text sampling during validation intervals to observe model convergence.
* **Configurable Entropy**: Temperature-controlled sampling () allows for adjusting the probability distribution of the next character.

### 2. Sentiment Classification (IMDb)

* **Sequence Processing**: Adapts the Transformer encoder to process variable-length movie reviews.
* **End-to-End Training**: Includes automated data downloading, tokenization, and evaluation metrics logging.

## Technical Stack

| Category | Tools/Technologies |
| --- | --- |
| **Frameworks** | PyTorch, Torchtext |
| **Monitoring** | Weights & Biases (W&B) |
| **Datasets** | enwik8 (Hutter Prize), IMDb |
| **Architecture** | Vanilla Transformer (Attention is All You Need) |

## Repository Structure

* `src/models/`: Core Transformer layers and architecture definitions.
* `src/training/`: Training loops, loss functions, and optimization logic.
* `scripts/`: High-level execution scripts for `generator.py` and `classifier.py`.
* `config/`: YAML/Argument-based hyperparameters for reproducible runs.

## Setup & Execution

### 1. Installation

```bash
git clone <repository-url>
pip install -r requirements.txt

```

### 2. Training the Generator

```bash
python scripts/generator.py --sample=True --visualize=True --temperature=0.7

```

### 3. Training the Classifier

```bash
python scripts/classifier.py --epochs=5 --batch_size=128 --visualize=True

```

---

## Visualizing Results

If `--visualize=True` is enabled, all metrics including validation accuracy, character-level loss, and gradient distributions will be synced to your W&B dashboard.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References & Credits
* **Architecture**: Vaswani, A., et al. "Attention Is All You Need" (2017).
* **Datasets**: Hutter Prize (enwik8) and Stanford AI Lab (IMDb).
* **Tools**: Experiment tracking by [Weights & Biases](https://wandb.ai).
