#------------------------------------------------------------------------------
#
# This is a guided project completed by following along with Andrej Karpathy
# Most code from "Let's build GPT: from scratch, in code, spelled out."
# https://www.youtube.com/watch?v=kCc8FmEb1nY&list=WL&index=51&t=769s
#
#------------------------------------------------------------------------------

import os
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
with open('tinyshksp.txt', 'r') as f:
    text = f.read()


# print("Length of dataset in characters: " , len(text), "\n")
# print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
#print(''.join(chars))
# print(vocab_size)

# Map characters to integers
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate (chars) }
encode = lambda s: [stoi[c] for c in s] # accepts string, outputs list of ints
decode = lambda l: "".join([itos[i] for i in l]) # takes in list of ints, outputs string

# print(encode("Hola"))
# print(decode(encode("boo")))

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])


# SPLITTING into train and validation sets

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    # print(f'When input is {context}, target {target}')

# 18:47
    
torch.manual_seed(1337)
batch_size = 4 # How many sequences are processed in parallel
block_size = 8 # Maximum length for the prediction

def get_batch(split):
    # Generate a set of inputs (x) and targets (y)
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Taking input and target tensors and stacking them, offsetting targets
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('Inputs: ')
print(xb.shape)
print(xb)
print('Targets: ')
print(yb.shape)
print(yb)

print("-----------------------------------")

for b in range(batch_size): # Iterating batches/going by tensor
    for t in range(block_size): # Iterating time/taking elements by tensor
        context = xb[b, :t+1]
        target = yb[b,t]
        # print(f"When input is {context.tolist()}, target is {target}")


# xb is the input to the transformer
print(xb)

# Implementing the bigram language model...

import torch.nn as nn
from torch.nn import functional as f
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Token reads logits for the next token using a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # idx and targets are (B,T) tensors of integers
        logits = self.token_embedding_table(idx) # (B, T, C)

        return logits
    
m = BigramLanguageModel(vocab_size)
out = m(xb, yb)
print(out.shape)

# 22:56

