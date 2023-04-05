import torch.nn as nn
from nlp_modules import TransformerEncoderRadford


class Transformer(nn.Module):
    def __init__(self, n_classes, max_sentence_size, latent_vs, nhead, mlp_dim):
        super(Transformer, self).__init__()

        self.pos_encoder = nn.Parameter(max_sentence_size, latent_vs)
        self.embedding = nn.Embedding(num_embeddings=max_sentence_size, embedding_dim=latent_vs)
        self.transformer = TransformerEncoderRadford(latent_vs=latent_vs, num_layers=12, nhead=nhead, mlp_dim=mlp_dim)

        self.fc = nn.Linear(latent_vs, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


