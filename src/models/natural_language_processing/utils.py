import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, word_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = torch.tensor(word_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask):
        mul = torch.bmm(q.transpose(1, 2), k) / torch.sqrt(self.dim)
        mul[mask] = -1000
        out = self.softmax(mul)
        print("out shape", out.shape)
        out = torch.bmm(out, v.transpose(1, 2))
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, nhead):
        super(MultiHeadSelfAttention, self).__init__()
        self.dim = dim
        self.nhead = nhead

        assert self.dim % self.nhead == 0, "dim must be a multiple of nhead"

        self.dim_proj = self.dim//self.nhead

        self.q_projs = [nn.Linear(self.dim, self.dim_proj) for _ in range(self.nhead)]
        self.k_projs = [nn.Linear(self.dim, self.dim_proj) for _ in range(self.nhead)]
        self.v_projs = [nn.Linear(self.dim, self.dim_proj) for _ in range(self.nhead)]

        self.heads = [ScaledDotProductAttention(dim=self.dim_proj) for _ in range(self.nhead)]

    def forward(self, q, k, v):
        out = torch.concat(torch.tensor([
            head(self.q_projs[i](q), self.k_projs[i](k), self.v_projs[i](v)) for i, head in enumerate(self.heads)
        ]))
        return out













