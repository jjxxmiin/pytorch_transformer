import torch
from transformer import Transformer_Classify, Transformer_Seq2seq

# Classification
d_model = 256
d_ff = 512
head = 8
N = 2
dropout = 0.1
vocab_size = 1000
target_size = 1000

model = Transformer_Classify(vocab_size,
                             d_model=d_model,
                             d_ff=d_ff,
                             head=head,
                             N=N,
                             dropout=dropout)

source = torch.rand(1, 1000).long()

print(model(source).shape)

# Seq2Seq
model = Transformer_Seq2seq(vocab_size,
                            target_size,
                            d_model=d_model,
                            d_ff=d_ff,
                            head=head,
                            N=N,
                            dropout=dropout)

source = torch.rand(1, 1000).long()
target = torch.rand(1, 1000).long()

print(model(source, target).shape)
