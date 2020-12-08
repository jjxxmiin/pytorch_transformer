import torch
import torch.nn as nn
from .modules import clones, LayerNorm, Sublayer1Output


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=None)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attention, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.slf_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer_output = clones(Sublayer1Output(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        x = self.sublayer_output[0](x, lambda x: self.slf_attention(x, x, x, mask))
        return self.sublayer_output[1](x, self.feed_forward)