import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, backbone):
        super(ImageEncoder, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()

