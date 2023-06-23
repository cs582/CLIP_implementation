import math
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.checkpoint import checkpoint
from src.models.natural_language_processing.nlp_modules import TransformerRadford

from src.models.natural_language_processing.nlp_token_embedding import TokenEmbedder

tokenizer = Tokenizer.from_file("src/data/nlp/tokenizers/CLIP-bpe.tokenizer.json")


class GPTSmall(nn.Module):
    def __init__(self, dim_out, vocab_size, max_length, use_checkpoint, device):
        super(GPTSmall, self).__init__()
        self.max_length = max_length
        self.device = device

        # The embedder takes size vocabulary_size because it should ignore the dummy token 0
        self.token_embedder = TokenEmbedder(vocabulary_size=vocab_size, embedding_dim=dim_out)
        self.transformer = TextTransformer(dim_model=dim_out, n_layers=8, max_length=max_length, nhead=8, dim_ff=1024, use_checkpoint=use_checkpoint)

    def forward(self, x):
        b, _ = x.shape

        # Creating masks
        mask = torch.zeros(b, self.max_length, self.max_length, device=self.device, dtype=torch.bool)
        eos_mask = torch.zeros(b, self.max_length, device=self.device, dtype=torch.bool)

        # Create masks
        eos_mask[:, :] = (x == tokenizer.token_to_id('[EOS]'))
        for small_b in range(b):
            sentence_length = (x[small_b] != 0).sum()
            mask[small_b, :sentence_length, :sentence_length] = torch.triu(torch.ones(sentence_length, sentence_length), diagonal=1).T

        # Token embedder
        x = self.token_embedder(x)
        # Transformer backbone
        x = self.transformer(x, mask, eos_mask)
        return x


class GPTBase(nn.Module):
    def __init__(self, dim_out,  vocab_size, max_length, use_checkpoint, device):
        super(GPTBase, self).__init__()
        self.max_length = max_length
        self.device = device

        # The embedder takes size vocabulary_size+1 because it should ignore the dummy token 0
        self.token_embedder = TokenEmbedder(vocabulary_size=vocab_size, embedding_dim=dim_out)
        self.transformer = TextTransformer(dim_model=dim_out, n_layers=12, max_length=max_length, nhead=8, dim_ff=2048, use_checkpoint=use_checkpoint)


    def forward(self, x):
        b, _ = x.shape

        # Creating masks
        mask = torch.zeros(b, self.max_length, self.max_length, device=self.device, dtype=torch.bool)
        eos_mask = torch.zeros(b, self.max_length, device=self.device, dtype=torch.bool)

        # Create masks
        eos_mask[:, :] = (x == tokenizer.token_to_id('[EOS]'))
        for small_b in range(b):
            sentence_length = (x[small_b] != 0).sum()
            mask[small_b, :sentence_length, :sentence_length] = torch.triu(torch.ones(sentence_length, sentence_length), diagonal=1).T

        # Token embedder
        x = self.token_embedder(x)
        # Transformer backbone
        x = self.transformer(x, mask, eos_mask)
        return x


class GPTLarge(nn.Module):
    def __init__(self, dim_out, vocab_size, max_length, use_checkpoint, device):
        super(GPTLarge, self).__init__()
        self.max_length = max_length
        self.device = device

        # Token Embedding
        self.token_embedder = TokenEmbedder(vocabulary_size=vocab_size, embedding_dim=dim_out)
        self.transformer = TextTransformer(dim_model=dim_out, n_layers=12, max_length=max_length, nhead=12, dim_ff=2048, use_checkpoint=use_checkpoint)

    def forward(self, x):
        b, _ = x.shape

        # Creating masks
        mask = torch.zeros(b, self.max_length, self.max_length, device=self.device, dtype=torch.bool)
        eos_mask = torch.zeros(b, self.max_length, device=self.device, dtype=torch.bool)

        # Create masks
        eos_mask[:, :] = (x == tokenizer.token_to_id('[EOS]'))
        for small_b in range(b):
            sentence_length = (x[small_b] != 0).sum()
            mask[small_b, :sentence_length, :sentence_length] = torch.triu(torch.ones(sentence_length, sentence_length), diagonal=1).T

        # Token embedder
        x = self.token_embedder(x)
        # Transformer backbone
        x = self.transformer(x, mask, eos_mask)
        return x


class TextTransformer(nn.Module):
    def __init__(self, n_layers, dim_model, max_length, dim_ff, nhead, use_checkpoint):
        super(TextTransformer, self).__init__()

        self.n_layers = n_layers
        self.nhead = nhead

        self.use_checkpoint = use_checkpoint

        self.dim_model = dim_model
        self.dim_ff = dim_ff

        self.max_length = max_length

        # Positional embedding
        self.pos_encoder = nn.Parameter(self._get_pos_encoding(max_length, dim_model))

        self.transformers = nn.ModuleList([
            TransformerRadford(dim_model=self.dim_model, nhead=self.nhead, dim_ff=self.dim_ff) for _ in range(self.n_layers)
        ])

        self.to_latent = nn.Identity()

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(self.dim_model)

        self._initialize_weights()

    def forward(self, x, mask, eos_mask): # b x l_max x dim_v
        # Add position encoder to token embedded input x
        x = torch.add(x, self.pos_encoder)

        # Transformer layers
        for l in range(self.n_layers):
            if self.use_checkpoint:
                x = checkpoint(self.transformers[l], x, mask)
            else:
                x = self.transformers[l](x, mask)

        # Get last [EOS] token at highest layer
        x = x[eos_mask]

        # Feature representation
        x = self.to_latent(x)
        x = self.layer_norm(x)
        return x

    def _get_pos_encoding(self, max_length, dim_model):
        pos = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * -(math.log(10000.0) / dim_model))
        pos_encoding = torch.zeros((1, max_length, dim_model))
        pos_encoding[0, :, 0::2] = torch.sin(pos * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(pos * div_term)
        return nn.Parameter(pos_encoding, requires_grad=False)

    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


