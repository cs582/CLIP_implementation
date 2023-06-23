import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CLIPModule(nn.Module):
    def __init__(self, image_encoder, text_encoder, dim_img, dim_text, embedding_dim, temperature):
        super(CLIPModule, self).__init__()
        # The embedding dimension is equal to the number of image-queries to pair in the training stage
        self.embedding_dim = embedding_dim

        self.dim_img = dim_img
        self.dim_text = dim_text

        self.temperature = torch.tensor(np.log(1 / temperature), requires_grad=True)

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.img_mm_encoder = nn.Linear(self.dim_img, self.embedding_dim, bias=False)
        self.txt_mm_encoder = nn.Linear(self.dim_text, self.embedding_dim, bias=False)

    def txt_encoder(self, x):
        x = self.text_encoder(x)
        x = self.txt_mm_encoder(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def img_encoder(self, x):
        x = self.image_encoder(x)
        x = self.img_mm_encoder(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, image, text):
        # Extract feature representation of each modality
        img_f = self.img_encoder(image) # batch_size x embedding_dim
        txt_f = self.txt_encoder(text)   # batch_size x embedding_dim

        # Scaled pairwise cosine similarities
        logits_images = torch.matmul(img_f, txt_f.t())  # batch_size x batch_size
        logits_images = logits_images * torch.clamp(torch.exp(self.temperature), min=0.01, max=100.0) # Scale by temperature
        logits_text = logits_images.t()

        return logits_images, logits_text











