import torch
import torch.nn as nn


class CLIPLoss(nn.Module):
    def __init__(self, temperature):
        super(CLIPLoss, self).__init__()
        self.temp = temperature

    def forward(self, img_e, txt_e):
        # Define loss for each task
        images_loss = nn.CrossEntropyLoss()
        text_loss = nn.CrossEntropyLoss()

        # Scaled pairwise cosine similarities
        logits = torch.dot(img_e, txt_e) * torch.exp(self.temp)

        # Loss function
        labels = torch.arange(0, len(img_e))
        loss_images = images_loss(logits, labels)
        loss_text = text_loss(logits.T, labels)
        loss = (loss_images + loss_text)/2
        return loss