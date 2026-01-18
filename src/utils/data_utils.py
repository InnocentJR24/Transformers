import torch
import gzip
import os
import wget
import pickle
import random
import numpy as np
from tqdm import tqdm

def here(subpath=None):
    """Return the absolute path to the project root or a subpath."""
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))

def enwik8(path=None, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """Load the enwik8 dataset from a gzip file or a local file."""
    if path is None:
        path = here('data/enwik8.gz')
    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

def sample_batch(data, length, batch_size, device):
    """Sample a batch of sequences from the dataset."""
    starts = torch.randint(low=0, high=data.size(0) - length - 1, size=(batch_size,))
    seq_inputs = [data[start:start+length] for start in starts]
    seq_targets = [data[start+1:start+length+1] for start in starts]
    inputs = torch.cat([s[None,:] for s in seq_inputs], dim=0).to(dtype=torch.long)
    targets = torch.cat([s[None,:] for s in seq_targets], dim=0).to(dtype=torch.long)
    return inputs.to(device), targets.to(device)

def load_imdb(final=False, val=5000, seed=0, voc=None, char=False):
    IMDB_URL = 'http://dlvu.github.io/data/imdb.{}.pkl.gz'
    IMDB_FILE = 'data/imdb.{}.pkl.gz'
    PAD, START, END, UNK = '.pad', '.start', '.end', '.unk'
    cst = 'char' if char else 'word'
    imdb_url = IMDB_URL.format(cst)
    imdb_file = IMDB_FILE.format(cst)

    if not os.path.exists(imdb_file):
        wget.download(imdb_url, out=f"data/")

    with gzip.open(imdb_file) as file:
        sequences, labels, i2w, w2i = pickle.load(file)

    if voc is not None and voc < len(i2w):
        nw_sequences = {}
        i2w = i2w[:voc]
        w2i = {w: i for i, w in enumerate(i2w)}
        mx, unk = voc, w2i['.unk']
        for key, seqs in sequences.items():
            nw_sequences[key] = [[s if s < mx else unk for s in seq] for seq in seqs]
        sequences = nw_sequences

    if final:
        return (sequences['train'], labels['train']), (sequences['test'], labels['test']), (i2w, w2i), 2

    random.seed(seed)
    x_train, y_train, x_val, y_val = [], [], [], []
    val_ind = set(random.sample(range(len(sequences['train'])), k=val))
    for i, (s, l) in enumerate(zip(sequences['train'], labels['train'])):
        if i in val_ind:
            x_val.append(s)
            y_val.append(l)
        else:
            x_train.append(s)
            y_train.append(l)

    return (x_train, y_train), (x_val, y_val), (i2w, w2i), 2

def variable_batches(x, y, w2i, size):
    """Create variable-length batches of sequences."""	
    batchLen = 0
    batch, batches, label, labels = [], [], [], []
    x, y = list(zip(*sorted(zip(x, y), key=lambda x: len(x[0]))))
    for i in tqdm(range(len(x)), desc="Batches", leave=False):
        if batchLen + len(x[i]) > size:
            if batch:
                for j in tqdm(range(len(batch)), desc="Padding", leave=False):
                    batch[j] += [w2i['.pad']] * (len(batch[-1]) - len(batch[j]))
                batches.append(torch.tensor(batch, dtype=torch.long))
                labels.append(torch.tensor(label))
                batch, label, batchLen = [], [], 0
        batch.append(x[i])
        label.append(y[i])
        batchLen += len(x[i])
    if batch:
        for j in tqdm(range(len(batch)), desc="Padding", leave=False):
            batch[j] += [w2i['.pad']] * (len(batch[-1]) - len(batch[j]))
        batches.append(torch.tensor(batch, dtype=torch.long))
        labels.append(torch.tensor(label))
    return batches, labels