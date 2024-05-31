#5/30/2024
#this code is made with the help of a video, and im going through this for learning purposes

#imports
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 500
max_iters = 5000
learning_rate = 3e-4
block_size = 256 #max number of tokens used as context
batch_size = 64 #how many independent sequences in parallel
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

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

train_data[:block_size+1]


"""x=train_data[:block_size]
y=train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")"""

torch.manual_seed(1337)

def get_batch(split):
    #gen small batch of data of inputs x and targets y
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')
"""print('inputs: ')
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
        print(f'When target is {context.tolist()} the target: {target}')"""

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
                             
    def forward(self,x):
        B, T, C= x.shape
        k = self.key(x)
        q = self.query(x)
        #compute attnetion scores/affinities
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        #wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T) from repo
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        #perform weighted aggregation of values
        v = self.value(x)
        out = wei @ v
        return out

class MultHeadAttention(nn.Module):
    #multiple heads of self attention in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        #linear layer followed by non-linearity
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        #n_embed: number of embedding dimesions, n_head: number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.sa_heads = MultHeadAttention(4, n_embed//4) #4 heads of 8-dimensional self-attention
        self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed,  vocab_size)

        # better init, not covered in the original GPT video, but important
        #self.apply(self._init_weights)

    """def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)"""

    def forward(self, idx, targets = None):
        B, T = idx.shape

        #idx and targets are both (B,T) tensor of integers
        token_embed = self.token_embedding_table(idx) # (B,T,C) batch, time, channel
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = token_embed + pos_embed
        x = self.blocks(x)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) #(B,T,vocab_size)

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
            #crop idx to the last block_size token
            idx_cond = idx[:, -block_size:]
            #get predicitons
            logits, loss = self(idx_cond)
            #ofucs only on last time step
            logits = logits[:, -1, :]
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)# (B,C)s
            #sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            #append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx

    
model = BigramLanguageModel()
m = model.to(device)

logits, loss = m(xb,yb)
print(logits.shape)
print(loss)

#prints garbage at this point
#print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

#optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for iter in range(max_iters):#train model, takes roughly 20 seconds
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    #evaluate the loss
    logit, loss  = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
#context = torch.zeros((1, 1), dtype=torch.long, device=device) #from repo
print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))


#----Code that exists for theory, not actually needed for generation or training-----
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
x.shape

#we want x[b,t] = mean_{i<=t} x{b,i}
xbow = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] #(t,C)
        xbow[b, t] = torch.mean(xprev, 0)

#xbow second version
wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x #(B, ,T) @ (B, T, C) ----> (B,T,C)
torch.allclose(xbow, xbow2)

#xbow 3rd version, determines affinities of elementes to one another using lower triangle
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)

# version 4: self attention
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) # (B,T,16)
q = query(x) # (B,T,16)
wei = q @ k.transpose(-2, -1) # (B,T,16) @ (B,16,T) ---> (B,T,T)

tril = torch.tril(torch.ones(T,T))
#wei = torch.zeros(T,T)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
v = value(x)
out = wei  @ v
out.shape
#--------------------------------------------------------------------