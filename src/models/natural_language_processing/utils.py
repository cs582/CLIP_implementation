import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, vector_dim, forward_dim):
        super(FeedForward, self).__init__()
        self.in_layer = nn.Linear(vector_dim, forward_dim, bias=True)
        self.out_layer = nn.Linear(forward_dim, vector_dim, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.in_layer(x))
        x = self.out_layer(x)
        return x


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

    def forward(self, x, mask):
        tensors = []
        for i in range(0, self.nhead):
            q_i, k_i, v_i = self.q_projs[i](x), self.k_projs[i](x), self.v_projs[i](x)
            head_i = self.heads[i](q_i, k_i, v_i, mask)
            tensors.append(head_i)
        out = torch.cat(tensors, dim=2)
        return out


class TransformerRadford(nn.Module):
    def __init__(self, embedding_dim, latent_vector_size, nhead, forward_dim):
        super(TransformerRadford, self).__init__()

        self.embedding_dim = embedding_dim
        self.latent_vector_size = latent_vector_size
        self.nhead = nhead
        self.forward_dim = forward_dim

        self.attention = MultiHeadSelfAttention(embedd_dim=embedding_dim, vector_size=latent_vector_size, nhead=nhead)
        self.ff = FeedForward(vector_dim=latent_vector_size, forward_dim=forward_dim)

        self.norm = nn.LayerNorm(normalized_shape=latent_vector_size)

    def forward(self, x, mask):
        a = self.attention(x, mask)
        a = self.norm(a) + a

        b = self.ff(a)
        b = self.norm(b) + b

        return b














