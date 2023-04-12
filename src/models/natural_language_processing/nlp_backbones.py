import torch
import torch.nn as nn
from src.models.natural_language_processing.nlp_modules import TransformerRadford

from collections import OrderedDict


class TextTransformer(nn.Module):
    def __init__(self, n_classes, layers, dim_model, max_length, dim_ff, nhead):
        super(TextTransformer, self).__init__()

        self.layers = layers
        self.nhead = nhead

        self.dim_model = dim_model
        self.dim_ff = dim_ff

        self.max_length = max_length

        self.tkn_embedding_encoder = nn.Parameter(torch.rand(1, self.dim_model, self.dim_model))
        self.pos_encoder = nn.Parameter(torch.rand(1, self.max_length, self.dim_model))

        self.transformer = nn.Sequential(OrderedDict(
            (f'layer{i}', TransformerRadford(dim_model=self.dim_model, nhead=self.nhead, dim_ff=self.dim_ff)) for i in range(self.layers)
        ))

        self.fc = nn.Linear(self.dim_model, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x): # b x l_max x dim_v
        x = self.token_embedding(x) # b x l_max x dim_v * dim_v x dim_emb -> b x l_max x dim_emb
        x += self.pos_encoder # b x l_max x dim_emb
        x = self.transformer(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


