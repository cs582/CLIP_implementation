import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask):
        mul = torch.bmm(q, k.transpose(1, 2)) / (len(q[0])**0.5)
        mul[mask] = -1000.0
        out = self.softmax(mul)
        out = torch.bmm(out, v)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedd_dim, vector_size, nhead):
        super(MultiHeadSelfAttention, self).__init__()
        self.dim = vector_size
        self.embedd_dim = embedd_dim
        self.nhead = nhead

        assert self.dim % self.nhead == 0, "dim must be a multiple of nhead"

        self.dim_proj = self.dim//self.nhead

        self.q_projs = [nn.Linear(self.embedd_dim, self.dim_proj) for _ in range(self.nhead)]
        self.k_projs = [nn.Linear(self.embedd_dim, self.dim_proj) for _ in range(self.nhead)]
        self.v_projs = [nn.Linear(self.embedd_dim, self.dim_proj) for _ in range(self.nhead)]

        self.heads = [ScaledDotProductAttention() for _ in range(self.nhead)]

    def forward(self, q, k, v, mask):
        tensors = []
        for i in range(0, self.nhead):
            q_i, k_i, v_i = self.q_projs[i](q), self.k_projs[i](k), self.v_projs[i](v)
            head_i = self.heads[i](q_i, k_i, v_i, mask)
            tensors.append(head_i)
        out = torch.cat(tensors, dim=2)
        return out













