import torch
import torch.nn as nn


class TransformerRadford(nn.Module):
    def __init__(self, dim_model, nhead, dim_ff):
        super(TransformerRadford, self).__init__()
        self.masked_self_attention = MaskedMultiHeadSelfAttention(dim_model=dim_model, n_head=nhead)
        self.layer_norm_att = nn.LayerNorm(dim_model)

        self.mlp = MultilayerPerceptron(dim_model=dim_model, dim_ff=dim_ff)
        self.layer_norm_mlp = nn.LayerNorm(dim_model)

    def forward(self, x, mask):
        x = torch.add(self.masked_self_attention(x, mask), x)
        x = self.layer_norm_att(x)
        x = torch.add(self.mlp(x), x)
        out = self.layer_norm_mlp(x)
        return out


class MultilayerPerceptron(nn.Module):
    def __init__(self, dim_model, dim_ff):
        super(MultilayerPerceptron, self).__init__()
        self.dim_model = dim_model
        self.dim_ff = dim_ff

        self.fc_hidden = nn.Linear(self.dim_model, self.dim_ff)
        self.fc_out = nn.Linear(self.dim_ff, self.dim_model)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_hidden(x))
        x = self.fc_out(x)
        return x


class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_model, n_head):
        super(MaskedMultiHeadSelfAttention, self).__init__()

        assert dim_model % n_head == 0, "dim_att must be a multiple of n_head"

        self.dim_model = dim_model
        self.dim_att = dim_model // n_head

        self.n_head = n_head

        self.attention_heads = [
            MaskedSelfAttention(dim_x=self.dim_model, dim_att=self.dim_att) for _ in range(self.n_head)
            ]

        self.fc = nn.Linear(self.dim_model, self.dim_model)

    def forward(self, x, mask):
        x_heads = [None] * self.n_head
        for i, head in enumerate(self.attention_heads):
            x_heads[i] = head(x, mask)
        x = torch.cat(x_heads, dim=2)
        x = self.fc(x)
        return x


class MaskedSelfAttention(nn.Module):
    def __init__(self, dim_x, dim_att):
        super(MaskedSelfAttention, self).__init__()

        self.dim_x = dim_x

        self.dim_k = dim_att
        self.dim_v = dim_att

        self.wq = nn.Parameter(torch.rand(1, self.dim_x, self.dim_k))
        self.wk = nn.Parameter(torch.rand(1, self.dim_x, self.dim_k))
        self.wv = nn.Parameter(torch.rand(1, self.dim_x, self.dim_v))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask=None): # b x l_max x dim_v
        q = torch.matmul(x, self.wq) # b x l_max x dim_q
        k = torch.matmul(x, self.wk) # b x l_max x dim_k
        v = torch.matmul(x, self.wv) # b x l_max x dim_v

        s = torch.bmm(q, k.transpose(1, 2)) # b x l_max x l_max
        s[~mask] = -1000.0
        s = s / (self.dim_k ** 0.5) # regularization
        s = self.softmax(s)
        out = torch.bmm(s, v)
        return out


