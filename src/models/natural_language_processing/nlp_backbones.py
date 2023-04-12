import torch
import torch.nn as nn
from src.models.natural_language_processing.nlp_modules import TransformerRadford

from collections import OrderedDict


class TextTransformer(nn.Module):
    def __init__(self, n_classes, n_layers, dim_model, max_length, dim_ff, nhead):
        super(TextTransformer, self).__init__()

        self.n_layers = n_layers
        self.nhead = nhead

        self.dim_model = dim_model
        self.dim_ff = dim_ff

        self.max_length = max_length

        self.tkn_embedding_encoder = nn.Parameter(torch.rand(1, self.dim_model, self.dim_model))
        self.pos_encoder = nn.Parameter(torch.rand(1, self.max_length, self.dim_model))

        self.transformers = [
            TransformerRadford(dim_model=self.dim_model, nhead=self.nhead, dim_ff=self.dim_ff) for _ in range(self.n_layers)
        ]

        self.to_latent = nn.Identity()

        self.fc = nn.Linear(self.dim_model, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask): # b x l_max x dim_v
        # Token embedding and position embedding
        x = torch.matmul(x, self.tkn_embedding_encoder)     # b x l_max x dim_v -> b x l_max x dim_v
        x = torch.add(x, self.pos_encoder)                  # b x l_max x dim_v

        # Transformer layers
        for l in range(self.n_layers):
            x = self.transformers[l](x, mask)

        # Get last [EOS] token
        x = x[:, mask.diff(dim=1)]
        x = self.to_latent(x)

        x = self.fc(x)
        x = self.softmax(x)
        return x


