import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolution1(nn.Module):
    def __init__(self):
        super(Convolution1, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=4)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class Convolution2(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(Convolution2, self).__init__()
        # Convolution 2 goes through max pooling already so dim is already reduced
        self.block1 = ConvolutionBlock(in_channels=in_channels, out_channels=out_channels)
        self.block2 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)
        self.block3 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class Convolution3(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super(Convolution3, self).__init__()
        self.block1 = ConvolutionBlock(in_channels=in_channels, out_channels=out_channels, first_block=True)
        self.block2 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)
        self.block3 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)
        self.block4 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class Convolution4(nn.Module):
    def __init__(self, in_channels=128, out_channels=256):
        super(Convolution4, self).__init__()
        self.block1 = ConvolutionBlock(in_channels=in_channels, out_channels=out_channels, first_block=True)
        self.block2 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)
        self.block3 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)
        self.block4 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)
        self.block5 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)
        self.block6 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x


class Convolution5(nn.Module):
    def __init__(self, in_channels=256, out_channels=512):
        super(Convolution5, self).__init__()
        self.block1 = ConvolutionBlock(in_channels=in_channels, out_channels=out_channels, first_block=True)
        self.block2 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)
        self.block3 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)
        self.block4 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)
        self.block5 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)
        self.block6 = ConvolutionBlock(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first_block=False):
        super(ConvolutionBlock, self).__init__()
        # If first block, then set stride = 2
        self.first_block = first_block

        initial_stride = 2 if self.first_block else 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=initial_stride)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.pool = BlurPool2d(n_channels=in_channels)
        if self.first_block:
            self.convB = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        if self.first_block: # Down-sampling only in first block
            # Path A
            dsA = self.conv1(x) # CNN Down-Sampling
            out1 = self.conv2(dsA)

            # Path B
            dsB = self.pool(x) # BlurPooling Down-Sampling
            out2 = self.convB(dsB)

            out = torch.add(out1, out2)
        else: # If not first block then proceed as usual
            out = self.conv1(x)
            out = self.conv2(out)
        return out


class BlurPool2d(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.blur_kernel = torch.tensor([ [1, 2, 1], [2, 4, 2], [1, 2, 1] ], dtype=torch.float32).expand(n_channels, 1, 3, 3)
        self.blur_kernel = self.blur_kernel / (16*n_channels)  # Normalize the blur kernel

    def forward(self, x):
        # Pad the input tensor with zeros so that the tensor can be divided evenly into 3x3 regions
        if x.shape[2] % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))
        if x.shape[3] % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0))

        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.conv2d(x, self.blur_kernel, stride=2, padding=1, groups=self.n_channels)

        return x
    
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, nhead, mlp_dim, vector_size):
        super(TransformerEncoderBlock, self).__init__()

        self.encoder1 = nn.TransformerEncoderLayer(d_model=vector_size, nhead=nhead, activation='gelu', dim_feedforward=mlp_dim)
        self.encoder2 = nn.TransformerEncoderLayer(d_model=vector_size, nhead=nhead, activation='gelu', dim_feedforward=mlp_dim)
        self.encoder3 = nn.TransformerEncoderLayer(d_model=vector_size, nhead=nhead, activation='gelu', dim_feedforward=mlp_dim)
        self.encoder4 = nn.TransformerEncoderLayer(d_model=vector_size, nhead=nhead, activation='gelu', dim_feedforward=mlp_dim)

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        return x

