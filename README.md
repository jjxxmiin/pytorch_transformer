# pytorch_transformer
Transformer using pytorch


![transformer](https://github.com/jjeamin/pytorch_transformer/blob/main/assets/transformer.PNG)

## Positional Encoding

- Location Infomation

1. First word : 0, Last word : 1
2. Assign a linearly number
3. Constant distance
4. Has a unique value

```
# Sinusoidal positional encoding

w_k = 1 / 10000 ** (2k / d)

f(t)^i  =   sin(w_k),  if i = 2k
            cos(w_k),  if i = 2k + 1
```

## Self Attention

- Look at words in different locations in the input sentence.

- Query : word at current location
- Key   : word in different location.
- Value : search for related word 

```
x = query * key 
x = x / sqrt(key size) # stable gradient
x = softmax(x)
x = x * value
```

## Multi Head Attention

## Reference

- [https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch](https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch)
- [https://www.youtube.com/watch?v=mxGCEWOxfe8](https://www.youtube.com/watch?v=mxGCEWOxfe8)
