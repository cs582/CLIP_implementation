import torch
import torch.nn as nn
from src.models.natural_language_processing.nlp_modules import TransformerRadford

from src.models.natural_language_processing.nlp_token_embedding import TokenEmbedder


class TransformerB(nn.Module):
    def __init__(self, dim_out, batch_size, vocab_size, max_length):
        super(TransformerB, self).__init__()
        self.batch_size = batch_size
        self.max_length = max_length

        self.token_embedder = TokenEmbedder(vocabulary_size=vocab_size, embedding_dim=dim_out)
        self.transformer = TextTransformer(dim_model=dim_out, n_layers=12, max_length=max_length, nhead=8, dim_ff=2048)

        # Setting as tensor buffer, not updated in backpropagation
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
        self.register_buffer('mask', mask)

    def forward(self, x):
        print("\nx in", x.shape)
        # Mask
        self.mask = (x != -1)
        # Set other tokens to zero
        x[~self.mask] = 0.0
        print("after setting -1 to 0.0", x.shape)
        # Token embedder
        x = self.token_embedder(x)
        print("after token embedding", x.shape)
        # Transformer backbone
        x = self.transformer(x, self.mask)
        print("after transformer", x.shape)
        return x


class TransformerL(nn.Module):
    def __init__(self, dim_out, batch_size, vocab_size, max_length):
        super(TransformerL, self).__init__()
        self.batch_size = batch_size
        self.max_length = max_length

        self.token_embedder = TokenEmbedder(vocabulary_size=vocab_size, embedding_dim=dim_out)
        self.transformer = TextTransformer(dim_model=dim_out, n_layers=12, max_length=max_length, nhead=12, dim_ff=2048)

        # Setting as tensor buffer, not updated in backpropagation
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
        self.register_buffer('mask', mask)

    def forward(self, x):
        print("\nx in", x.shape)
        # Mask
        self.mask = (x != -1)
        # Set other tokens to zero
        x[~self.mask] = 0.0
        print("after setting -1 to 0.0", x.shape)
        # Token embedder
        x = self.token_embedder(x)
        print("after token embedding", x.shape)
        # Transformer backbone
        x = self.transformer(x, self.mask)
        print("after transformer", x.shape)
        return x


class TextTransformer(nn.Module):
    def __init__(self, n_layers, dim_model, max_length, dim_ff, nhead):
        super(TextTransformer, self).__init__()

        self.n_layers = n_layers
        self.nhead = nhead

        self.dim_model = dim_model
        self.dim_ff = dim_ff

        self.max_length = max_length

        self.tkn_embedding_encoder = nn.Parameter(torch.rand(1, self.dim_model, self.dim_model))
        self.pos_encoder = nn.Parameter(torch.rand(1, self.max_length, self.dim_model))

        self.transformers = nn.ModuleList([
            TransformerRadford(dim_model=self.dim_model, nhead=self.nhead, dim_ff=self.dim_ff) for _ in range(self.n_layers)
        ])

        self.to_latent = nn.Identity()

        self.fc = nn.Linear(self.dim_model, self.dim_model)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask): # b x l_max x dim_v
        b, _, _ = x.shape

        # Token embedding and position embedding
        x = torch.matmul(x, self.tkn_embedding_encoder)     # b x l_max x dim_v -> b x l_max x dim_v
        x = torch.add(x, self.pos_encoder)                  # b x l_max x dim_v

        # Transformer layers
        for l in range(self.n_layers):
            x = self.transformers[l](x, mask)

        # Get last [EOS] token
        print("pre last word", x.shape)
        print("mask", mask)
        last_word = torch.cat((mask, torch.zeros(b, 1, dtype=torch.bool, device=mask.device)), dim=1).diff(dim=1) # last word mask
        print("last word", last_word)
        print("last word mask", last_word.shape)
        x = x[last_word]
        print("x after", x.shape)
        x = self.to_latent(x)

        x = self.fc(x)
        x = self.softmax(x)
        return x


