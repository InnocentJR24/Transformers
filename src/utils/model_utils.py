import torch
import torch.nn.functional as F
import torch.distributions as dist

def sample(probdist, temperature):
    """Sample a character from a probability distribution."""
    if temperature == 0:
        return torch.argmax(probdist)
    adjdist = F.softmax(probdist / temperature, dim=0)
    char = dist.Categorical(adjdist).sample()
    return char

def sample_sequence(seed, model, length, sample_length, temperature):
    """Generate a sequence using the model."""
    print("[", end="", flush=True)
    for char in seed:
        print(str(chr(char)), end="", flush=True)
    print("]", end="", flush=True)

    sequence = seed.detach().clone()
    
    for _ in range(sample_length):
        input = sequence[-length:]
        output = model(input[None, :])
        char = sample(output[0, -1], temperature=temperature)
    
        print(str(chr(max(32, char))), end="", flush=True)
        sequence = torch.cat((sequence, char[None]), dim=0)

    print()
    return seed

def positionEncoding(length, embedding_dim):
    """Generate positional encoding for a given sequence length and embedding dimension."""
    n = 10000
    pe = torch.zeros(length, embedding_dim)
    for pos in range(length):
        for i in range(0, embedding_dim, 2):
            theta = pos / (n ** ((2 * i) / embedding_dim))
            pe[pos, i] = torch.sin(torch.tensor(theta))
            pe[pos, i + 1] = torch.cos(torch.tensor(theta))
    return pe

def mask_(data):
    """Apply causal mask to attention weights."""
    b, t, k = data.size()
    mask = torch.triu_indices(t, t, offset=1)
    data[..., mask[0], mask[1]] = float('-inf')
    return data

def load_model(path):
    """Load a saved model from a file."""
    if path is None:
        print("No path provided. Please provide a valid model path.")
        return
    model = torch.load(path, weights_only=False)
    model.eval()
    print(f"Model loaded from {path}")
    return model

def save_model(model, save_path, seed, length, depth, embedding_dim, num_batches, batch_size):
    """Save the model to a file."""
    if save_path is None:
        save_path = f"models/saved/model_s{seed}_l{length}-d{depth}-e{embedding_dim}-nb{num_batches}-bs{batch_size}.pt"
    torch.save(model, save_path)
    print(f"Model saved as {save_path}")