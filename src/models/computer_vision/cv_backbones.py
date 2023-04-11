import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from src.models.computer_vision.cv_modules import Convolution1, Convolution2, Convolution3, Convolution4, Convolution5, TransformerEncoderBlock


class RN34_at224(nn.Module):
    def __init__(self, embedding_dim):
        super(RN34_at224, self).__init__()

        # Convolutional Layers
        # 224 x 224
        self.conv1 = Convolution1() # 113 x 113 - > 56 x 56
        self.conv2 = Convolution2() # 56 x 56
        self.conv3 = Convolution3() # 28 x 28
        self.conv4 = Convolution4() # 14 x 14
        self.conv5 = Convolution5() # 7 x 7

        # Final Stage
        # 7 x 7 x 1024 -> 7*7*1024 -> embedding_dim
        self.avg_pool = nn.AvgPool2d(kernel_size=3)
        self.attention = nn.TransformerEncoderLayer(d_model=7*7*1024, nhead=8)
        self.fc = nn.Linear(7*7*1024, embedding_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolutional Stage
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Sixth stage
        x = self.avg_pool(x)
        x = x.view(-1, 1)
        x = self.attention(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class RN34_at336(nn.Module):
    def __init__(self, embedding_dim):
        super(RN34_at336, self).__init__()

        # Convolutional Layers
        # 336 x 336
        self.conv1 = Convolution1() # 166 x 166 -> 83 x 83
        self.conv2 = Convolution2() # 83 x 83
        self.conv3 = Convolution3() # 42 x 42
        self.conv4 = Convolution4() # 21 x 21
        self.conv5 = Convolution5() # 11 x 11

        # Final Stage
        # 11 x 11 x 1024 -> 11*11*1024 -> embedding_dim
        self.avg_pool = nn.AvgPool2d(kernel_size=3)
        self.attention = nn.TransformerEncoderLayer(d_model=11*11*1024, nhead=8)
        self.fc = nn.Linear(11*11*1024, embedding_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolutional Stage
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Sixth stage
        x = self.avg_pool(x)
        x = x.view(-1, 1)
        x = self.attention(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


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