import torch.nn as nn

class TransformerEncoderRadford(nn.Module):
    def __init__(self, latent_vs, nhead, mlp_dim, num_layers):
        super(TransformerEncoderRadford, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_vs, nhead=nhead, dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, mask_check=True)

    def forward(self, x, mask):
        return self.transformer(x, mask)

