# Transformer Text Generation & Classification

This repository contains a Transformer-based implementation for character-level text generation and classification, trained on the enwik8 or IMDb dataset. It includes scripts for training a text generator and a classifier, with support for Weights & Biases (W&B) visualization.

## Overview

- **Text Generation**: Trains a Transformer model on the enwik8 dataset to generate character-level text. The generator can be sampled to produce readable text output, allowing you to see predicted text in real-time during training.
- **Classification**: Trains a Transformer classifier on the IMDb dataset for sentiment analysis.
- **Features**: Configurable parameters, gradient visualization, model saving/loading, W&B logging, and text sampling for the generator.

## Repository Structure

- `config/`: Configuration files for model and training settings.
- `data/`: Dataset files (e.g., `enwik8.gz` for generation).
- `models/saved/`: Saved model checkpoints.
- `scripts/`: Training scripts (`generator.py`, `classifier.py`).
- `src/models/`: Model definitions.
- `src/utils/`: Data and model utility functions.
- `src/training/`: Training helper functions.
- `.gitignore`: Git ignore file.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependencies.

## Setup

1. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Ensure `data/enwik8.gz` exists for text generation (download from http://mattmahoney.net/dc/textdata if needed).
   - IMDb dataset is loaded automatically via the script.

4. **Set Up Weights & Biases** (optional):
   - Install W&B: `pip install wandb`
   - Log in: `wandb login`

## Usage

### Running

1. **Default Training**:
   ```bash
   python scripts/generator.py
   python scripts/classifier.py
   ```

2. **Customize Parameters**:
   - Generator:
     ```bash
     python scripts/generator.py --num_batches=1000 --batch_size=64 --grad_vis=True
     ```
   - Classifier:
     ```bash
     python scripts/classifier.py --epochs=5 --batch_size=5000
     ```

3. **Enable W&B Logging**:
   ```bash
   python scripts/generator.py --visualize=True
   python scripts/classifier.py --visualize=True
   ```

### Sampling the Generator

The generator supports text sampling to preview predicted output during training. To enable sampling:
- Use the `--sample=True` flag to generate sample text at validation intervals.
- Adjust `--sample_length` to set the length of the generated text (default: 512 characters).
- Control output creativity with `--temperature` (default: 0.5; lower values make output more deterministic, higher values increase randomness).

Example:
```bash
python scripts/generator.py --num_batches=1000 --sample=True --sample_length=1024 --temperature=0.7 --visualize=True
```
- During training, sampled text will be printed to the console or logged to W&B (if `--visualize=True`) at every validation interval (controlled by `--val_interval`).
- The output reflects the model's current ability to predict coherent text based on the enwik8 dataset, improving as training progresses.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Transformer architecture based on "Attention Is All You Need" (Vaswani et al., 2017)
- enwik8 dataset from the Hutter Prize
- IMDb dataset via torchtext/Hugging Face datasets
- Experiment tracking powered by [Weights & Biases](https://wandb.ai)
