import torch.nn as nn


class TransformerRadford(nn.Module):
    def __init__(self, latent_vs, nhead, mlp_dim):
        super(TransformerRadford, self).__init__()
        self.transformer = nn.TransformerEncoder(mask_check=True)

    def forward(self, x):
        return self.transformer(x)


class TransformerBlock(nn.Module):
    def __init__(self, latent_vs, nhead, mlp_dim):
        super(TransformerBlock, self).__init__()

        self.enc1 = TransformerRadford(latent_vs=latent_vs, nhead=nhead, mlp_dim=mlp_dim)
        self.enc2 = TransformerRadford(latent_vs=latent_vs, nhead=nhead, mlp_dim=mlp_dim)
        self.enc3 = TransformerRadford(latent_vs=latent_vs, nhead=nhead, mlp_dim=mlp_dim)
        self.enc4 = TransformerRadford(latent_vs=latent_vs, nhead=nhead, mlp_dim=mlp_dim)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return x

