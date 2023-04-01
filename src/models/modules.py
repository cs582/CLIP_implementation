import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolution1(nn.Module):
    def __init__(self):
        super(Convolution1, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class ConvolutionX(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ConvolutionX, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)

        self.pool = BlurPool2d()
        self.res = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)

        x2 = self.pool(x)
        x2 = self.res(x2)
        return x1 + x2


class BlurPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blur_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        self.blur_kernel = self.blur_kernel.view(1, 1, 3, 3) / 16.0  # Normalize the blur kernel

    def forward(self, x):
        # Pad the input tensor with zeros so that the tensor can be divided evenly into 2x2 regions
        if x.shape[2] % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))
        if x.shape[3] % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0))

        # Blur the input tensor using the Gaussian kernel
        x = F.conv2d(x, self.blur_kernel, stride=2, padding=1, groups=x.shape[1])

        # Apply max pooling with 2x2 kernel and stride of 2
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        return x