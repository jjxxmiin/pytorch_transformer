from transformer import Transformer

d_model = 256
d_ff = 512
head = 8
N = 2
dropout = 0.1
vocab_size = 1000

model = Transformer(vocab_size,
                    d_model=d_model,
                    d_ff=d_ff,
                    head=head,
                    N=N,
                    dropout=dropout)

print(model)