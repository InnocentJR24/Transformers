import torch
import torch.nn as nn
from .modules import TransformerBlock

class TransformerGenerator(nn.Module):
    def __init__(self, num_char, embedding_dim, pe, heads, hidden_mult, depth):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=num_char, embedding_dim=embedding_dim)
        self.pe = pe
        
        tblocks = []
        for _ in range(depth):
            tblocks.append(TransformerBlock(embedding_dim=embedding_dim, heads=heads, hidden_mult=hidden_mult))

        self.tblocks = nn.Sequential(*tblocks)
        self.toprob = nn.Linear(embedding_dim, num_char)

    def forward(self, x):
        x = self.embedding(x)
        b, t, e = x.size()

        x = x + self.pe.unsqueeze(0)
        x = self.tblocks(x)

        y = x.view(b * t, e)
        y = self.toprob(y)
        y = y.view(b, t, -1)
        return y
    
class TransformerClassifier(nn.Module):
    def __init__(self, w2i, numcls, embedding_dim, pe, heads, hidden_mult, depth):
        super().__init__()

        self.embedding = nn.Embedding(len(w2i), embedding_dim, padding_idx=w2i['.pad'])
        self.pe = pe
        
        tblocks = []
        for _ in range(depth):
            tblocks.append(TransformerBlock(embedding_dim=embedding_dim, heads=heads, hidden_mult=hidden_mult))

        self.tblocks = nn.Sequential(*tblocks)
        self.linear = nn.Linear(embedding_dim, numcls)

    def forward(self, x):
        x = self.embedding(x)
        b, t, k = x.size()

        x = x + self.pe[:t].unsqueeze(0)
        x = self.tblocks(x)

        y = torch.mean(x, dim=1)
        y = self.linear(y)
        return y