import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


# ViT (Dosovitskiy et. al. 2020)
class ViT(nn.Module):
    def __init__(self, patch_resolution, img_size, n_classes):
        super(ViT, self).__init__()
        self.h, self.w = img_size
        self.p = patch_resolution

        self.n_channels = 3

        # Number of patches
        self.n_embeddings = (self.h * self.w) // (self.p**2)

        # Latent vector size D
        vector_size = 512

        nhead = 16
        mlp_size = 2048

        # Get patches
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p, p2=self.p)

        # Patch embedding
        self.patch_embedding_encoder = nn.Parameter(torch.randn(1, self.n_embeddings, vector_size))
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, vector_size))
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_embeddings + 1, vector_size))

        # Layer normalization as in CLIP (Radford et al. 2021)
        self.layer_norm = nn.LayerNorm(vector_size)

        # Transformer Encoder Hidden Layers
        self.t_encoder1 = TransformerEncoderBlock(vector_size=vector_size, nhead=nhead, mlp_dim=mlp_size)
        self.t_encoder2 = TransformerEncoderBlock(vector_size=vector_size, nhead=nhead, mlp_dim=mlp_size)
        self.t_encoder3 = TransformerEncoderBlock(vector_size=vector_size, nhead=nhead, mlp_dim=mlp_size)
        self.to_latent = nn.Identity()

        # Out MLP head with one
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(vector_size),
            nn.Linear(vector_size, mlp_size),
            nn.Linear(mlp_size, n_classes),
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