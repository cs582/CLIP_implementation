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
    def __init__(self, n_channels=64):
        super(Convolution2, self).__init__()
        # Convolution 2 goes through max pooling already so dim is already reduced
        self.block1 = ConvolutionBlock(in_channels=n_channels, out_channels=n_channels)
        self.block2 = ConvolutionBlock(in_channels=n_channels, out_channels=n_channels)
        self.block3 = ConvolutionBlock(in_channels=n_channels, out_channels=n_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class Convolution3(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super(Convolution3, self).__init__()
        self.block1 = ConvolutionBlock(in_channels=in_channels, out_channels=out_channels, downsampling=True)
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
        self.block1 = ConvolutionBlock(in_channels=in_channels, out_channels=out_channels, downsampling=True)
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
        self.block1 = ConvolutionBlock(in_channels=in_channels, out_channels=out_channels, downsampling=True)
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
    def __init__(self, in_channels, out_channels, downsampling=False):
        super(ConvolutionBlock, self).__init__()
        # If first block, then set stride = 2
        self.downsampling = downsampling

        initial_stride = 2 if self.downsampling else 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=initial_stride)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.pool = BlurPool2d(n_channels=in_channels)
        if self.downsampling:
            self.convB = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        if self.downsampling: # Down-sampling only in first block
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


class BlurPool2d(nn.Module):
    def __init__(self, n_channels):
        super(BlurPool2d, self).__init__()
        self.n_channels = n_channels
        self.blur_kernel = torch.tensor([ [1, 2, 1], [2, 4, 2], [1, 2, 1] ], dtype=torch.float32).expand(n_channels, 1, 3, 3)
        self.blur_kernel = self.blur_kernel / 16  # Normalize the blur kernel
        self.blur_kernel.to(BlurPool2d.get_device())

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


class AttentionPooling(nn.Module):
    def __init__(self, in_size, dim_attention=32):
        super(AttentionPooling, self).__init__()
        self.vk_size = int(np.prod(in_size))
        self.q_size = int(np.prod(np.add(in_size, -2)))

        self.dim_attention = dim_attention

        self.pool = nn.AvgPool2d(kernel_size=3, stride=1)

        self.wq = nn.Parameter(torch.rand(1, self.q_size, self.dim_attention))
        self.wk = nn.Parameter(torch.rand(1, self.vk_size, self.dim_attention))
        self.wv = nn.Parameter(torch.rand(1, self.vk_size, self.dim_attention))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_pool = self.pool(x)

        x = x.flatten(start_dim=2)              # batch_size x n_channels x vk_dim
        x_pool = x_pool.flatten(start_dim=2)    # batch_size x n_channels x q_dim

        q = torch.matmul(x_pool, self.wq)  # b x c x q_dim * dim_q x dim_att -> b x c x dim_att
        k = torch.matmul(x, self.wk)       # b x c x vk_dim * dim_vk x dim_att -> b x c x dim_att
        v = torch.matmul(x, self.wv)       # b x c x vk_dim * dim_vk x dim_att -> b x c x dim_att

        s = torch.bmm(q, k.transpose(1, 2))     # (b x) c x dim_att * (b x) dim_att x c -> (b x) c x c
        s = s / (self.dim_attention ** 0.5)     # regularization
        s = self.softmax(s)                     # activation function

        v_hat = torch.bmm(s, v)                 # b x c x c
        return v_hat








