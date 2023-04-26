import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPModule(nn.Module):
    def __init__(self, image_encoder, text_encoder, dim_img, dim_text, embedding_dim, temperature):
        super(CLIPModule, self).__init__()
        # The embedding dimension is equal to the number of image-queries to pair in the training stage
        self.embedding_dim = embedding_dim

        self.dim_img = dim_img
        self.dim_text = dim_text

        self.temperature = torch.tensor(temperature, requires_grad=True)

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.img_mm_encoder = nn.Parameter(torch.randn(self.dim_img, self.embedding_dim))
        self.txt_mm_encoder = nn.Parameter(torch.randn(self.dim_text, self.embedding_dim))

    def forward(self, image, text):
        # Extract feature representation of each modality
        img_f = self.image_encoder(image) # batch_size x dim_img
        txt_f = self.text_encoder(text)   # batch_suze x dim_text

        # Joint multimodal embedding
        img_e = torch.matmul(img_f, self.img_mm_encoder) # batch_size x dim_emb
        img_e = F.normalize(img_e, p=2, dim=1) # l2 normalization

        txt_e = torch.matmul(txt_f, self.txt_mm_encoder) # batch_size x dim_emb
        txt_e = F.normalize(txt_e, p=2, dim=1) # l2 normalization

        # Max value of the logits
        clip_upper = self.temperature * 100

        # Scaled pairwise cosine similarities
        logits = torch.matmul(img_e, txt_e.transpose(0,1))  # batch_size x batch_size
        logits = torch.maximum(logits, clip_upper) * torch.exp(self.temperature) # Scale by temerature
        return logits











