import torch
import torch.nn as nn

class CLIPModule(nn.Module):
    def __init__(self, image_encoder, text_encoder, dim_img, dim_text, embedding_dim, temperature):
        super(CLIPModule, self).__init__()
        # The embedding dimension is equal to the number of image-queries to pair in the training stage
        self.embedding_dim = embedding_dim

        self.dim_img = dim_img
        self.dim_text = dim_text

        self.temperature = temperature

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.img_mm_encoder = nn.Parameter(torch.randn(self.dim_img, self.embedding_dim))
        self.txt_mm_encoder = nn.Parameter(torch.randn(self.dim_text, self.embedding_dim))

    def forward(self, image, text):
        # Extract feature representation of each modality
        img_f = self.image_encoder(image) # batch_size x dim_img
        txt_f = self.text_encoder(text)   # batch_suze x dim_text

        # Joint multimodal embedding
        img_e = torch.norm(torch.matmul(img_f, self.img_mm_encoder), p=2, dim=1) # batch_size
        txt_e = torch.norm(torch.matmul(txt_f, self.txt_mm_encoder), p=2, dim=1) # batch_size

        # Scaled pairwise cosine similarities
        logits = torch.dot(img_e, txt_e) * torch.exp(self.temperature) # batch_size x batch_size
        return logits











