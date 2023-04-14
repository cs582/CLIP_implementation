import torch
import torch.nn as nn


class TokenEmbedder(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim):
        super(TokenEmbedder, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)