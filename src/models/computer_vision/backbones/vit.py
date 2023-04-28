import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


# ViT (Dosovitskiy et. al. 2020) / CLIP (Radford et al. 2021)

# ViT-B/32 @ 224
class ViTBat224(nn.Module):
    def __init__(self, dim_out):
        super(ViTBat224, self).__init__()
        # ViT Hyper-parameters
        self.c, self.h, self.w = (3, 224, 224)
        self.p = 32

        # Number of layers
        self.n_layers = 12
        # Latent vector size D
        self.vector_size = 768
        # FF dim
        self.mlp_size = 4096
        # Number of heads
        self.nhead = 12

        # Number of patches
        self.n_embeddings = (self.h * self.w) // (self.p**2)

        # Get patches
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p, p2=self.p)

        # Patch embedding
        self.patch_embedding = nn.Parameter(torch.randn(1, (self.p**2)*self.c, self.vector_size))
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, self.vector_size))
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_embeddings + 1, self.vector_size))

        # Layer normalization as in CLIP (Radford et al. 2021)
        self.layer_norm = nn.LayerNorm(self.vector_size)

        # Transformer Encoder Hidden Layers
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.vector_size, activation='gelu', nhead=self.nhead, dim_feedforward=self.mlp_size, norm_first=True, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=self.n_layers)
        self.to_latent = nn.Identity()

        # Out MLP head with one
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.vector_size),
            nn.Linear(self.vector_size, self.mlp_size),
            nn.Linear(self.mlp_size, dim_out),
        )

    def forward(self, x):
        N, c, h, w = x.shape

        # Convert image to patches
        x = self.rearrange(x)

        # Patch + Position Embedding
        x = torch.matmul(x, self.patch_embedding)
        class_token = self.class_token.expand(N, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x += self.pos_embedding
        x = self.layer_norm(x)

        # Transformer Encoder Layers
        x = self.transformer(x)

        # Getting class token
        x = x[:, 0]
        x = self.to_latent(x)

        x = self.mlp_head(x)
        return x

# ViT-L/14 @ 224
class ViTLat224(nn.Module):
    def __init__(self, dim_out):
        super(ViTLat224, self).__init__()
        # ViT Hyper-parameters
        self.c, self.h, self.w = (3, 224, 224)
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
        self.patch_embedding = nn.Parameter(torch.randn(1, (self.p**2)*self.c, self.vector_size))
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, self.vector_size))
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_embeddings + 1, self.vector_size))

        # Layer normalization as in CLIP (Radford et al. 2021)
        self.layer_norm = nn.LayerNorm(self.vector_size)

        # Transformer Encoder Hidden Layers
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.vector_size, activation='gelu', nhead=self.nhead, dim_feedforward=self.mlp_size, norm_first=True, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=self.n_layers)
        self.to_latent = nn.Identity()

        # Out MLP head with one
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.vector_size),
            nn.Linear(self.vector_size, self.mlp_size),
            nn.Linear(self.mlp_size, dim_out),
        )

    def forward(self, x):
        N, c, h, w = x.shape

        # Convert image to patches
        x = self.rearrange(x) # b x 256 x 588

        # Patch + Position Embedding
        x = torch.matmul(x, self.patch_embedding)           # b x 256 x 588 * 588 x D -> b x 256 x D
        class_token = self.class_token.expand(N, -1, -1)    # b x 1 x D
        x = torch.cat((class_token, x), dim=1)              # cat(b x 256 x D, b x 1 x D) -> b x 257 x D
        x += self.pos_embedding                             # 257 x D
        x = self.layer_norm(x)

        # Transformer Encoder Layers
        x = self.transformer(x)

        # Getting class token
        x = x[:, 0]
        x = self.to_latent(x)

        x = self.mlp_head(x)
        return x
