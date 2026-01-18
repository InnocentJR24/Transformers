import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.model_utils import mask_
import math

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, heads):
        super().__init__()

        assert embedding_dim % heads == 0

        self.toKey = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.toQuery = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.toValue = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.unifyHeads = nn.Linear(embedding_dim, embedding_dim)
        self.heads = heads
    
    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        s = k // h

        keys = self.toKey(x)
        queries = self.toQuery(x)
        values = self.toValue(x)

        keys = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values = values.view(b, t, h, s)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        w = torch.bmm(queries, keys.transpose(1, 2))
        w = w / math.sqrt(s)
        w = mask_(w)
        w = F.softmax(w, dim=2)

        y = torch.bmm(w, values)
        y = y.view(b, h, t, s)
        y = y.transpose(1, 2).contiguous().view(b, t, s * h)
        y = self.unifyHeads(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, heads, hidden_mult):
        super().__init__()

        self.attention = SelfAttention(embedding_dim, heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feedForward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * hidden_mult),
            nn.ReLU(),
            nn.Linear(embedding_dim * hidden_mult, embedding_dim)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = attended + x
        x = self.norm1(x)
        fedforward = self.feedForward(x)
        x = fedforward + x
        x = self.norm2(x)
        return x