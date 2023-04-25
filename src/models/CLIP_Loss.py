import torch
import torch.nn as nn


class CLIPLoss(nn.Module):
    def __init__(self, logits_length):
        super(CLIPLoss, self).__init__()
        self.logits_length = logits_length
        self.register_buffer("labels", torch.arange(0, self.logits_length))

    def forward(self, logits):
        # Define loss for each task
        images_loss = nn.CrossEntropyLoss()
        text_loss = nn.CrossEntropyLoss()

        # Loss function
        loss_images = images_loss(logits, self.labels)
        loss_text = text_loss(logits.transpose(0, 1), self.labels)
        loss = (loss_images + loss_text)/2
        return loss