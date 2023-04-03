import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from modules import Convolution1, ConvolutionX, TransformerEncoderBlock


class Resnet50D_CLIP(nn.Module):
    def __init__(self, n_classes):
        super(Resnet50D_CLIP, self).__init__()
        # Stage 1
        self.conv1 = Convolution1()

        # Stage 2
        self.conv2_1 = ConvolutionX(in_channels=64, hidden_channels=64, out_channels=64)
        self.conv2_2 = ConvolutionX(in_channels=64, hidden_channels=64, out_channels=64)
        self.conv2_3 = ConvolutionX(in_channels=64, hidden_channels=64, out_channels=256)

        # Stage 3
        self.conv3_1 = ConvolutionX(in_channels=256, hidden_channels=128, out_channels=128)
        self.conv3_2 = ConvolutionX(in_channels=128, hidden_channels=128, out_channels=128)
        self.conv3_3 = ConvolutionX(in_channels=128, hidden_channels=128, out_channels=128)
        self.conv3_4 = ConvolutionX(in_channels=128, hidden_channels=128, out_channels=512)

        # Stage 4
        self.conv4_1 = ConvolutionX(in_channels=512, hidden_channels=256, out_channels=256)
        self.conv4_2 = ConvolutionX(in_channels=256, hidden_channels=256, out_channels=256)
        self.conv4_3 = ConvolutionX(in_channels=256, hidden_channels=256, out_channels=256)
        self.conv4_4 = ConvolutionX(in_channels=256, hidden_channels=256, out_channels=256)
        self.conv4_5 = ConvolutionX(in_channels=256, hidden_channels=256, out_channels=256)
        self.conv4_6 = ConvolutionX(in_channels=256, hidden_channels=256, out_channels=1024)

        # Stage 5
        self.conv5_1 = ConvolutionX(in_channels=1024, hidden_channels=512, out_channels=512)
        self.conv5_2 = ConvolutionX(in_channels=512, hidden_channels=512, out_channels=512)
        self.conv5_3 = ConvolutionX(in_channels=512, hidden_channels=512, out_channels=2048)

        # Final Stage
        self.avg_pool = nn.AvgPool2d(kernel_size=3)
        self.attention = nn.TransformerEncoderLayer(d_model=112*112*1024, nhead=8)
        self.fc = nn.Linear(112*112*1024, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # First stage
        x = self.conv1(x)

        # Second stage
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        # Third stage
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        # Fourth stage
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        # Fifth stage
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        # Sixth stage
        x = self.avg_pool(x)
        x = x.view(-1, 1)
        x = self.attention(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

# ViT (Dosovitskiy et. al. 2020)
class ViT_CLIP_Large(nn.Module):
    def __init__(self, patch_resolution, img_size, n_classes):
        super(ViT_CLIP_Large, self).__init__()
        self.h, self.w = img_size
        self.p = patch_resolution

        self.n_channels = 3

        # Number of patches
        self.n_embeddings = (self.h * self.w) // (self.p**2)

        # Latent vector size D
        vector_size = 1024

        nhead = 16
        mlp_size = 4096

        # Get patches
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p, p2=self.p)

        # Embedding encoder
        self.patch_embedding_encoder = nn.Parameter(torch.randn(1, self.n_embeddings, vector_size))
        # Class embedding
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