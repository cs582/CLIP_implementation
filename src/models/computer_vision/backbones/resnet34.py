import torch
import torch.nn as nn

from src.models.computer_vision.cv_modules import Convolution1, Convolution2, Convolution3, Convolution4, Convolution5, TransformerEncoderBlock, AttentionPooling


class RN34at224(nn.Module):
    def __init__(self, embedding_dim):
        super(RN34at224, self).__init__()

        # Convolutional Layers
        # 224 x 224
        self.conv1 = Convolution1() # 113 x 113 - > 56 x 56
        self.conv2 = Convolution2() # 56 x 56
        self.conv3 = Convolution3() # 28 x 28
        self.conv4 = Convolution4() # 14 x 14
        self.conv5 = Convolution5() # 7 x 7

        # Final Stage
        self.attention_pooling = AttentionPooling(in_size=(7,7), dim_attention=32)
        self.fc = nn.Linear(32*512, embedding_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolutional Stage
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Sixth stage
        x = self.attention_pooling(x)
        x = self.fc(x.flatten(start_dim=1))
        x = self.softmax(x)
        return x


class RN34at336(nn.Module):
    def __init__(self, embedding_dim):
        super(RN34at336, self).__init__()

        # Convolutional Layers
        # 336 x 336
        self.conv1 = Convolution1() # 166 x 166 -> 83 x 83
        self.conv2 = Convolution2() # 83 x 83
        self.conv3 = Convolution3() # 42 x 42
        self.conv4 = Convolution4() # 21 x 21
        self.conv5 = Convolution5() # 11 x 11

        # Final Stage
        self.attention_pooling = AttentionPooling(in_size=(11,11), dim_attention=32)
        self.fc = nn.Linear(32*512, embedding_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolutional Stage
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Sixth stage
        x = self.attention_pooling(x)
        x = self.fc(x.flatten(start_dim=1))
        x = self.softmax(x)
        return x
