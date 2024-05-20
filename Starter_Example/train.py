#5/17/2024
#this code is made with the help of a video, and im going through this for learning purposes

#imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# read it in to inspect it
with open('GPT_Summer/Starter_Example/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x=train_data[:block_size]
y=train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

torch.manual_seed(1337)
batch_size = 4 #how many independent sequences will be running in parallel
block_size = 8 #max context length for predictions

def get_batch(split):
    #gen small batch of data of inputs x and targets y
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs: ')
print(xb.shape)
print(xb)
print('targets: ')
print(yb.shape)
print(yb)
print('-------')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f'When target is {context.tolist()} the target: {target}')

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        #each token directly reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        #idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) batch, time, channel
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) #loss should roughly be -ln(1/65) where 65 = vocab_size
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indeces in the current context
        for _ in range(max_new_tokens):
            #get predicitons
            logits, loss = self(idx)
            #ofucs only on last time step
            logits = logits[:, -1, :]
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)# (B,C)
            #sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            #append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx

    
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb,yb)
print(logits.shape)
print(loss)