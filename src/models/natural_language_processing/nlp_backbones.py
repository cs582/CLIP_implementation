import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from src.models.natural_language_processing.nlp_modules import TransformerRadford

from src.models.natural_language_processing.nlp_token_embedding import TokenEmbedder

additional_tokens = {"[SOS]": 43001, "[EOS]": 43000}


class TransformerB(nn.Module):
    def __init__(self, dim_out, batch_size, vocab_size, max_length):
        super(TransformerB, self).__init__()
        self.batch_size = batch_size
        self.max_length = max_length

        self.additional_tokens = additional_tokens

        # The embedder takes size vocabulary_size+1 because it should ignore the dummy token 0
        self.token_embedder = TokenEmbedder(vocabulary_size=vocab_size+1, embedding_dim=dim_out)
        self.transformer = TextTransformer(dim_model=dim_out, n_layers=12, max_length=max_length, nhead=8, dim_ff=2048)

        self.register_buffer('mask', torch.zeros(self.batch_size, self.max_length, self.max_length, dtype=torch.bool))
        self.register_buffer('eos_mask', torch.zeros(self.batch_size, self.max_length, dtype=torch.bool))

    def forward(self, x):
        b, _ = x.shape

        # Create masks
        self.mask[:, :, :] = 0.0
        self.eos_mask[:, :] = (x == self.additional_tokens["[EOS]"])
        for small_b in range(b):
            sentence_length = (x[small_b] != 0).sum()
            self.mask[small_b, :sentence_length, :sentence_length] = torch.triu(torch.ones(sentence_length, sentence_length), diagonal=1).T
        # Token embedder
        x = self.token_embedder(x)
        # Transformer backbone
        x = self.transformer(x, self.mask, self.eos_mask)
        return x


class TransformerL(nn.Module):
    def __init__(self, dim_out, batch_size, vocab_size, max_length):
        super(TransformerL, self).__init__()
        self.batch_size = batch_size
        self.max_length = max_length

        self.additional_tokens = additional_tokens

        # Token Embedding
        self.token_embedder = TokenEmbedder(vocabulary_size=vocab_size, embedding_dim=dim_out)
        self.transformer = TextTransformer(dim_model=dim_out, n_layers=12, max_length=max_length, nhead=12, dim_ff=2048)

        self.register_buffer('mask', torch.zeros(self.batch_size, self.max_length, self.max_length, dtype=torch.bool))
        self.register_buffer('eos_mask', torch.zeros(self.batch_size, self.max_length, dtype=torch.bool))

    def forward(self, x):
        b, _ = x.shape

        # Create masks
        self.mask[:, :, :] = 0.0
        self.eos_mask[:, :] = (x == self.additional_tokens["[EOS]"])
        for small_b in range(b):
            sentence_length = (x[small_b] != 0).sum()
            self.mask[small_b, :sentence_length, :sentence_length] = torch.triu(torch.ones(sentence_length, sentence_length), diagonal=1).T
        # Token embedder
        x = self.token_embedder(x)
        # Transformer backbone
        x = self.transformer(x, self.mask)
        return x


class TextTransformer(nn.Module):
    def __init__(self, n_layers, dim_model, max_length, dim_ff, nhead):
        super(TextTransformer, self).__init__()

        self.n_layers = n_layers
        self.nhead = nhead

        self.dim_model = dim_model
        self.dim_ff = dim_ff

        self.max_length = max_length

        # Intermediate encoder COMMENTED FOR TESTING
        #self.tkn_embedding_encoder = nn.Parameter(torch.rand(1, self.dim_model, self.dim_model))
        # Positional embedding
        self.pos_encoder = nn.Parameter(torch.rand(1, self.max_length, self.dim_model))

        self.transformers = nn.ModuleList([
            TransformerRadford(dim_model=self.dim_model, nhead=self.nhead, dim_ff=self.dim_ff) for _ in range(self.n_layers)
        ])

        self.to_latent = nn.Identity()

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(self.dim_model)
        # self.fc = nn.Linear(self.dim_model, self.dim_model)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask, eos_mask): # b x l_max x dim_v
        # Token embedding and position embedding

        # INTERMEDIATE ENCODER COMMENTED FOR TESTING
        #x = torch.matmul(x, self.tkn_embedding_encoder)     # b x l_max x dim_v -> b x l_max x dim_v
        x = torch.add(x, self.pos_encoder)                  # b x l_max x dim_v

        # Transformer layers
        for l in range(self.n_layers):
            x = checkpoint(self.transformers[l], x, mask)

        # Get last [EOS] token at highest layer
        x = x[eos_mask]

        # Feature representation
        x = self.to_latent(x)
        x = self.layer_norm(x)

        # Last forward and softmax commented for TESTING
        # x = self.fc(x)
        # x = self.softmax(x)
        return x


