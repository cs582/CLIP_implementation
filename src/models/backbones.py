import torch
import torch.nn as nn
from modules import Convolution1, ConvolutionX


class Resnet50D(nn.Module):
    def __init__(self, n_classes):
        super(Resnet50D, self).__init__()
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


class ViT(nn.Module):
    def __init__(self, in_size, n_channels, n_classes, dropout, vector_size, nhead=1):
        super(ViT, self).__init__()
        n_embeddings, row_dim = in_size

        self.patch_embedding_encoder = nn.Parameter(torch.randn(1, row_dim, vector_size))
        self.class_token = nn.Parameter(torch.randn(1, 1, vector_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_embeddings + 1, vector_size))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=vector_size, activation='gelu', nhead=nhead, dim_feedforward=vector_size),
            nn.TransformerEncoderLayer(d_model=vector_size, activation='gelu', nhead=nhead, dim_feedforward=vector_size),
            nn.TransformerEncoderLayer(d_model=vector_size, activation='gelu', nhead=nhead, dim_feedforward=vector_size),
            nn.TransformerEncoderLayer(d_model=vector_size, activation='gelu', nhead=nhead, dim_feedforward=vector_size)
        )

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(vector_size),
            nn.Linear(vector_size, n_classes)
        )

    def forward(self, x):
        N, n_channels, n_vectors, vector_dim = x.shape

        # Since this considers the case when n_channels = 1, simply reshape
        x = x.view(N, n_vectors, vector_dim)

        # Patch Embedding
        x = torch.matmul(x, self.patch_embedding_encoder)
        class_token = self.class_token.expand(N, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        # Transformer Encoder Layers
        x = self.transformer(x)

        # Getting class token
        x = x[:, 0]
        x = self.to_latent(x)

        x = self.mlp_head(x)
        return x