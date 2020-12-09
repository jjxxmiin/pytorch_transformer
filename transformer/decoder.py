import torch.nn as nn
from .modules import clones, LayerNorm, Sublayer1Output, Sublayer2Output


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, dec_input, enc_input, mask=None):
        for layer in self.layers:
            x = layer(dec_input, enc_input, mask=mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, slf_attention, enc_attention, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.slf_attention = slf_attention
        self.enc_attention = enc_attention
        self.feed_forward = feed_forward
        self.sublayer_output = nn.ModuleList([Sublayer1Output(size, dropout), Sublayer2Output(size, dropout), Sublayer1Output(size, dropout)])
        self.size = size

    def forward(self, dec_input, enc_output, mask=None):
        dec_output = self.sublayer_output[0](dec_input, lambda x: self.slf_attention(x, x, x, mask))
        dec_output = self.sublayer_output[1](dec_output, enc_output, lambda x1, x2: self.enc_attention(x1, x2, x2, None))
        return self.sublayer_output[2](dec_output, self.feed_forward)