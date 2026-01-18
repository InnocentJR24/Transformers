import torch
import wandb
import random
from src.utils.data_utils import enwik8, sample_batch
from src.utils.model_utils import positionEncoding, sample_sequence
from src.models.transformer import TransformerGenerator

def setup_seed(seed):
    """Set random seed for reproducibility."""
    if seed < 0:
        seed = random.randint(0, 1000000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return seed

def load_data(final, data_path="data/enwik8.gz"):
    """Load and prepare enwik8 dataset."""
    train, val, test = enwik8(path=data_path)
    if final:
        train = torch.cat([train, val], dim=0)
    else:
        test = val
    return train, test

def initialize_model_and_optimizer(num_char, length, embedding_dim, heads, hidden_mult, depth, lr, warmup_iter, start_factor, device):
    """Set up model, loss function, optimizer, and scheduler."""
    pe = positionEncoding(length=length, embedding_dim=embedding_dim).to(device)
    model = TransformerGenerator(
        num_char=num_char, embedding_dim=embedding_dim, pe=pe, 
        heads=heads, hidden_mult=hidden_mult, depth=depth
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer, start_factor=start_factor, total_iters=warmup_iter
    )
    return model, loss_fn, optimizer, scheduler

def log_metrics(batch_idx,  loss, bpb, prefix=""):
    """Log metrics to wandb or console."""
    metrics = {"Batch": batch_idx + 1, f"{prefix}Loss": loss, f"{prefix}Validation Bits per Byte": bpb}
    if wandb.run is not None:
        wandb.log(metrics)
    else:
        print(f"Batch {batch_idx+1}, {prefix}Loss: {loss:.4f}, {prefix}Validation Bits per Byte: {bpb:.2f}")

def log_gradients(model, batch_idx, log_individual):
    """Log total and individual gradient norms."""
    total_sq_norm = 0.0
    individual_norms = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            total_sq_norm += grad_norm.item() ** 2
            if log_individual:
                individual_norms[f"grad_norm/{name}"] = grad_norm.item()

    total_norm = total_sq_norm ** 0.5

    # Log total norm
    if wandb.run is not None:
        wandb.log({"Total Gradient Norm": total_norm}, step=batch_idx + 1)
        if log_individual:
            wandb.log(individual_norms, step=batch_idx + 1)
    else:
        print(f"Batch {batch_idx + 1}, Total Gradient Norm: {total_norm:.4f}")
        if log_individual:
            for key, value in individual_norms.items():
                print(f"{key}: {value:.6f}")


def validate_model(model, val_data, length, batch_size, loss_fn, num_char, device, sample, sample_length, temperature):
    """Perform validation and return bits per byte."""
    model.eval()
    x_val, y_val = sample_batch(data=val_data, length=length, batch_size=batch_size, device=device)
    with torch.no_grad():
        logits = model(x_val)
        loss = loss_fn(logits.view(-1, num_char), y_val.view(-1))
        bpb = loss.item() / torch.log(torch.tensor(2.0))
    
    if sample:
        seed = x_val[0]
        sample_sequence(seed=seed, model=model, length=length, sample_length=sample_length, temperature=temperature)
    
    model.train()
    return bpb