import torch
import torch.nn as nn
import fire
import wandb
from tqdm import tqdm

# Puts the project root directory in the system path for module imports
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.utils.data_utils import load_imdb
from src.utils.data_utils import variable_batches
from src.models.transformer import TransformerClassifier
from src.utils.model_utils import positionEncoding

def train(
    epochs=3,                   # Number of training epochs
    embedding_dim=256,          # Dimensionality of token embeddings
    heads=4,                    # Number of attention heads in each transformer block
    hidden_mult=4,              # Multiplier for hidden layer size
    depth=6,                    # Number of transformer blocks (layers)
    batch_size=10000,           # Number of characters in each training batch
    lr=0.0001,                  # Learning rate
    voc=10002,                  # Vocabulary size
    val_size=1000,              # Size of validation dataset
    seed=0,                     # Random seed for reproducibility
    visualize=False             # If True, log training metrics with Weights & Biases; else print to console
):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load data
    (x_train, y_train), (x_valid, y_valid), (i2w, w2i), numcls = load_imdb(final=False, val=val_size, seed=seed, voc=voc)
    
    # Assertions
    assert isinstance(x_train, list)
    assert len(x_train) == len(y_train)

    # Create batches
    x_batch, y_batch = variable_batches(x_train, y_train, w2i, size=batch_size)
    xv_batch, yv_batch = variable_batches(x_valid, y_valid, w2i, size=batch_size)
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pe = positionEncoding(length=len(x_batch[-1][0]), embedding_dim=embedding_dim).to(device)
    model = TransformerClassifier(w2i=w2i, numcls=numcls, embedding_dim=embedding_dim, pe=pe, heads=heads, hidden_mult=hidden_mult, depth=depth).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Initialize wandb for visualization if requested
    if visualize:
        wandb.init(project="imdb_classifier", config={
            "epochs": epochs, "embedding_dim": embedding_dim, "batch_size": batch_size,
            "lr": lr, "voc": voc, "val_size": val_size, "seed": seed
        })

    # Training loop
    for e in tqdm(range(epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0      
        for i in tqdm(range(len(x_batch)), desc="Training", leave=False):
            optimizer.zero_grad()
            inputs = x_batch[i].to(device)
            targets = y_batch[i].to(device)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for i in tqdm(range(len(xv_batch)), desc="Validation", leave=False):
                inputs = xv_batch[i].to(device)
                targets = yv_batch[i].to(device)
                logits = model(inputs).to(device)
                predictions = torch.argmax(logits, dim=1)
                total_correct += torch.sum(predictions == targets).item()
                total_samples += len(predictions)
        
        avg_loss = running_loss / len(x_batch)
        val_acc = (total_correct / total_samples) * 100

        if wandb.run is not None:
            wandb.log({"epoch": e + 1, "train_loss": avg_loss, "val_accuracy": val_acc})
        else:
            print(f"Epoch {e+1}, Train Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    fire.Fire(train)