import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    """Scaled Dot-Product Attention"""
    d_k = query.size(-1)
    scale_factor = 1/math.sqrt(d_k)

    score = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    if mask is not None:
        score = score.masked_fill(mask == 0, -1e9)
    
    attention_score = F.softmax(score)

    if dropout is not None:
        attention_score = dropout(attention_score)

    return torch.matmul(attention_score, value), attention_score


class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_k = self.d_v = d_model // head
        self.head = head

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.attention_score = None

        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        num_batch = q.size(0)

        if mask is not None:
            mask = mask.unsqueeze(1)

        query = self.wq(q).view(num_batch, -1, self.head, self.d_k).transpose(1, 2)
        key = self.wk(k).view(num_batch, -1, self.head, self.d_k).transpose(1, 2)
        value = self.wv(v).view(num_batch, -1, self.head, self.d_v).transpose(1, 2)

        x, self.attention_score = attention(query, key, value, mask=self.mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(num_batch, -1, self.head * self.d_k)
        x = self.wo(x)

        return x 