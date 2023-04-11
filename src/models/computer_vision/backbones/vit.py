import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


# ViT (Dosovitskiy et. al. 2020) / CLIP (Radford et al. 2021)
# ViT-L/14 @ 224
class ViTat224(nn.Module):
    def __init__(self, embedding_dim):
        super(ViTat224, self).__init__()
        # ViT Hyper-parameters
        self.n_channels, self.h, self.w = (3, 224, 224)
        self.p = 14

        # Number of layers
        self.n_layers = 24
        # Latent vector size D
        self.vector_size = 1024
        # FF dim
        self.mlp_size = 4096
        # Number of heads
        self.nhead = 16

        # Number of patches
        self.n_embeddings = (self.h * self.w) // (self.p**2)

        # Get patches
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p, p2=self.p)

        # Patch embedding
        self.patch_embedding_encoder = nn.Parameter(torch.randn(1, self.n_embeddings, self.vector_size))
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, self.vector_size))
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_embeddings + 1, self.vector_size))

        # Layer normalization as in CLIP (Radford et al. 2021)
        self.layer_norm = nn.LayerNorm(self.vector_size)

        # Transformer Encoder Hidden Layers
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.vector_size, activation='gelu', nhead=self.nhead, dim_feedforward=self.mlp_size)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=self.n_layers)
        self.to_latent = nn.Identity()

        # Out MLP head with one
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.vector_size),
            nn.Linear(self.vector_size, self.mlp_size),
            nn.Linear(self.mlp_size, embedding_dim),
        )

    def forward(self, x):
        N, c, h, w = x.shape

        # Convert image to patches
        x = self.rearrange(x)

        # Patch Embedding
        x = torch.matmul(x, self.patch_embedding_encoder)
        class_token = self.class_token.expand(N, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x += self.pos_embedding
        x = self.layer_norm(x)

        # Transformer Encoder Layers
        x = self.t_encoder1(x)
        x = self.t_encoder2(x)
        x = self.t_encoder3(x)

        # Getting class token
        x = x[:, 0]
        x = self.to_latent(x)

        x = self.mlp_head(x)
        return x


# ViT-L/14 @ 336
class ViTat336(nn.Module):
    def __init__(self, embedding_dim):
        super(ViTat336, self).__init__()
        # ViT Hyper-parameters
        self.n_channels, self.h, self.w = (3, 336, 336)
        self.p = 14

        # Number of layers
        self.n_layers = 24
        # Latent vector size D
        self.vector_size = 1024
        # FF dim
        self.mlp_size = 4096
        # Number of heads
        self.nhead = 16

        # Number of patches
        self.n_embeddings = (self.h * self.w) // (self.p**2)

        # Get patches
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p, p2=self.p)

        # Patch embedding
        self.patch_embedding_encoder = nn.Parameter(torch.randn(1, self.n_embeddings, self.vector_size))
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, self.vector_size))
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_embeddings + 1, self.vector_size))

        # Layer normalization as in CLIP (Radford et al. 2021)
        self.layer_norm = nn.LayerNorm(self.vector_size)

        # Transformer Encoder Hidden Layers
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.vector_size, activation='gelu', nhead=self.nhead, dim_feedforward=self.mlp_size)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=self.n_layers)
        self.to_latent = nn.Identity()

        # Out MLP head with one
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.vector_size),
            nn.Linear(self.vector_size, self.mlp_size),
            nn.Linear(self.mlp_size, embedding_dim),
        )

    def forward(self, x):
        N, c, h, w = x.shape

        # Convert image to patches
        x = self.rearrange(x)

        # Patch Embedding
        x = torch.matmul(x, self.patch_embedding_encoder)
        class_token = self.class_token.expand(N, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x += self.pos_embedding
        x = self.layer_norm(x)

        # Transformer Encoder Layers
        x = self.t_encoder1(x)
        x = self.t_encoder2(x)
        x = self.t_encoder3(x)

        # Getting class token
        x = x[:, 0]
        x = self.to_latent(x)

        x = self.mlp_head(x)
        return x