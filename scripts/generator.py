import torch
import fire
import wandb
import yaml
from tqdm import tqdm

# Puts the project root directory in the system path for module imports
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.utils.data_utils import sample_batch
from src.utils.model_utils import load_model, save_model
from src.training.train_utils import setup_seed, load_data, initialize_model_and_optimizer, log_metrics, log_gradients, validate_model
from torch.nn.utils import clip_grad_norm_

def train(
    config=None,                 # Placeholder for future configuration implementation
    num_batches=170000,         # Total number of training batches
    val_interval=1000,          # Run validation every `val_interval` batches
    val_mult=2,                 # Validation batch size multiplier
    batch_size=32,              # Number of sequences per training batch
    length=256,                 # Length of each input sequence
    embedding_dim=256,          # Dimensionality of token embeddings
    depth=12,                   # Number of transformer blocks
    heads=16,                    # Number of attention heads in each block
    hidden_mult=8,              # Multiplier for hidden layer size 
    lr=0.0003,                  # Base learning rate
    warmup_mult=0.05,           # Proportion of total batches used for learning rate warm-up
    start_factor=0.1,           # Starting LR factor 
    max_norm=1,                 # Maximum norm for gradient clipping
    sample=False,               # If True, generate sample output during training
    sample_length=512,          # Length of generated sample output
    temperature=0.5,            # Sampling temperature 
    log_individual=False,       # If True, log individual layer gradients
    grad_vis=True,              # If True, log gradient visualizations
    save=True,                  # If True, save the model periodically
    save_path=None,             # Path to save the trained model
    load_path=None,             # Path to load a pre-trained model
    seed=-1,                    # Random seed (-1 means use random seed)
    final=False,                # If True, validate on test set; otherwise use validation set
    visualize=True              # If True, log to Weights & Biases; else use print statements
):

    """Train a Transformer model on the enwik8 dataset."""
    # Load config file if provided
    if config is not None:
        pass

    num_char = 256  # Number of characters in dataset

    # Validate inputs
    assert embedding_dim % heads == 0, "Embedding dimension must be divisible by number of heads"
    assert num_batches % val_interval == 0, "Number of batches must be divisible by validation interval"

    # Setup seed and device
    seed = setup_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_data, val_data = load_data(final)

    # Initialize model and training components
    model, loss_fn, optimizer, scheduler = initialize_model_and_optimizer(
        num_char=num_char, length=length, embedding_dim=embedding_dim, heads=heads, hidden_mult=hidden_mult, depth=depth, lr=lr, warmup_iter=num_batches * warmup_mult, start_factor=start_factor, device=device
    )

    # Load pre-trained model if requested
    if load_path != None:
        model = load_model(path=load_path)
        bpb = validate_model(
                model=model, val_data=val_data, length=length, batch_size=batch_size * val_mult, loss_fn=loss_fn, num_char=num_char, device=device, 
                sample=sample, sample_length=sample_length, temperature=temperature
            )
        log_metrics(batch_idx=0, loss=0.0, bpb=bpb, prefix="Loaded ")
        return

    # Initialize wandb if visualization is enabled
    if visualize:
        wandb.init(project="Wikipedia Generator", config={
            "Batches": num_batches, "Validation Interval": val_interval,
            "Batch Size": batch_size, "Sequence Length": length,
            "Embedding Dimension": embedding_dim, "Transformer Blocks": depth,
            "Learning Rate": lr, "Warmup Multiplier": warmup_mult,
            "Start Factor": start_factor, "Attention Heads": heads,
            "Hidden Layers": hidden_mult, "Validation Size Multiplier": val_mult,
            "Max Norm": max_norm, "Seed": seed
        })
    else:
        print(f"Training for {num_batches} batches with seed {seed}...")

    # Training loop
    model.train()
    running_loss = 0.0
    for batch_idx in tqdm(range(num_batches), desc="Training"):
        x_train, y_train = sample_batch(
            data=train_data, length=length, batch_size=batch_size, device=device
        )

        # Forward and backward pass
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits.view(-1, num_char), y_train.view(-1))
        loss.backward()

        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=max_norm)

        # Log gradients at validation intervals
        if grad_vis and (batch_idx + 1) % val_interval == 0:
            log_gradients(model, batch_idx, log_individual)

        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        # Validation
        if (batch_idx + 1) % val_interval == 0:
            bpb = validate_model(
                model=model, val_data=val_data, length=length, batch_size=batch_size * val_mult, loss_fn=loss_fn, num_char=num_char, device=device, 
                sample=sample, sample_length=sample_length, temperature=temperature
            )
            batch_loss = running_loss / val_interval
            log_metrics(batch_idx=batch_idx, loss=batch_loss, bpb=bpb)
            running_loss = 0.0

    # Final logging
    print(f"Training complete. Seed: {seed}, Batches: {num_batches}, Final Loss: {batch_loss:.4f}, Final Bits per Byte: {bpb:.2f}")
    if wandb.run is not None:
        wandb.finish()

    # Save model if requested
    if save:
        save_model(
            model=model, save_path=save_path, seed=seed, length=length,
            depth=depth, embedding_dim=embedding_dim, num_batches=num_batches, batch_size=batch_size
        )

if __name__ == "__main__":
    fire.Fire(train)