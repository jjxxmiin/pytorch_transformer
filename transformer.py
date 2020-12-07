import torch
import torch.nn as nn
from modules import PositionalEncoding, Embedding
from attention import MultiHeadAttention
from feed_forward import FeedForward
from encoder import EncoderLayer, Encoder
from copy import deepcopy


class Transformer(nn.Module):
    def __init__(self, 
                 vocab_size,
                 d_model = 256,
                 d_ff = 512,
                 head = 8,
                 N = 1, 
                 dropout = 0.1):
        super(Transformer, self).__init__()

        input_embedding = Embedding(d_model, vocab_size)
        position_encoding = PositionalEncoding(d_model, dropout=dropout)
        self_attention = MultiHeadAttention(head, d_model, dropout=dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        encoder_layer = EncoderLayer(d_model, deepcopy(self_attention), deepcopy(feed_forward), dropout=dropout)

        self.embedder = nn.Sequential(input_embedding, 
                                      deepcopy(position_encoding))
        self.encoder = Encoder(encoder_layer, N)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedder(x.permute(1, 0))
        x = self.encoder(x)

        x = x[:,-1,:]
        out = self.fc(x)

        return out