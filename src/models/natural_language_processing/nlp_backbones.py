import torch
import torch.nn as nn
from nlp_modules import TransformerBlock

class NLP(nn.Module):
    def __init__(self, latent_vs, nhead, mlp_dim):
        super(NLP, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=49152, embedding_dim=latent_vs)

        self.block1 = TransformerBlock(latent_vs=latent_vs, nhead=nhead, mlp_dim=mlp_dim)
        self.block2 = TransformerBlock(latent_vs=latent_vs, nhead=nhead, mlp_dim=mlp_dim)
        self.block3 = TransformerBlock(latent_vs=latent_vs, nhead=nhead, mlp_dim=mlp_dim)

