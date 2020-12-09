import torch
import torch.nn as nn
from .modules import PositionalEncoding, Embedding
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder
from copy import deepcopy


def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return subsequent_mask


class Transformer_Classify(nn.Module):
    def __init__(self, 
                 vocab_size,
                 d_model = 256,
                 d_ff = 512,
                 head = 8,
                 N = 1, 
                 dropout = 0.1):
        super(Transformer_Classify, self).__init__()

        input_embedding = Embedding(d_model, vocab_size)
        position_encoding = PositionalEncoding(d_model, dropout=dropout)
        slf_attention = MultiHeadAttention(head, d_model, dropout=dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        encoder_layer = EncoderLayer(d_model, deepcopy(slf_attention), deepcopy(feed_forward), dropout=dropout)

        self.embedder = nn.Sequential(input_embedding, 
                                      deepcopy(position_encoding))
        self.encoder = Encoder(encoder_layer, N)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # batch first
        x = self.embedder(x)

        # batch last
        # x = self.embedder(x.permute(1, 0))

        x = self.encoder(x)

        x = x[:,-1,:]
        out = self.fc(x)

        return out


class Transformer_Seq2seq(nn.Module):
    def __init__(self, 
                 vocab_size,
                 target_size,
                 d_model = 256,
                 d_ff = 512,
                 head = 8,
                 N = 1, 
                 dropout = 0.1):
        super(Transformer_Seq2seq, self).__init__()

        embedding = Embedding(d_model, vocab_size)
        position_encoding = PositionalEncoding(d_model, dropout=dropout)

        slf_attention = MultiHeadAttention(head, d_model, dropout=dropout)
        enc_attention = MultiHeadAttention(head, d_model, dropout=dropout)

        feed_forward = FeedForward(d_model, d_ff, dropout=dropout)

        encoder_layer = EncoderLayer(d_model, deepcopy(slf_attention), deepcopy(feed_forward), dropout=dropout)
        decoder_layer = DecoderLayer(d_model, deepcopy(slf_attention), deepcopy(enc_attention), deepcopy(feed_forward), dropout=dropout)

        self.input_embedder = nn.Sequential(deepcopy(embedding), 
                                            deepcopy(position_encoding))
        self.encoder = Encoder(encoder_layer, N)

        self.output_embedder = nn.Sequential(deepcopy(embedding), 
                                             deepcopy(position_encoding))
        self.decoder = Decoder(decoder_layer, N)
        
        self.trg_word_prj = nn.Linear(d_model, target_size, bias=False)

    def forward(self, source, target):
        mask = get_attn_decoder_mask(target)
        
        # batch last
        # source = self.embedder(source.permute(1, 0))
        # target = self.embedder(target.permute(1, 0))

        source = self.input_embedder(source)
        enc_output = self.encoder(source, mask=None)

        target = self.output_embedder(target)
        dec_output = self.decoder(target, enc_output, mask=mask)

        output = self.trg_word_prj(dec_output)

        return output