import torch
import mmap
import random
import pickle
import io
import argparse
parser = argparse.ArgumentParser(description='Train a GPT model.')
parser.add_argument("-bs", type=int, required=True, help="select batch size")
args = parser.parse_args()
print(f"Batch size selected: {args.bs}")
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

block_size = args.bs
batch_size = 128
max_iters = 3000
learning_rate = 3e-4
eval_iter = 250
n_embd = 384
n_layer = 1
n_head = 1
dropout = 0.2

chars = ""
with open ('Mdm_Bovary.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

string_to_int = { ch:i for i,ch in enumerate (chars) }
int_to_string = { i:ch for i,ch in enumerate (chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join(int_to_string[i] for i in l)

def get_random_chunk(split):
    filename = 'train_data.txt' if split == 'train' else 'val_data.txt'
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size-1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', ' ')
            data = torch.tensor(encode(text), dtype=torch.long)
    return data

def get_batch(split): #decide whether to use train or val data
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        #compute attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5 #(B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #mask out future positions
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        wei = self.dropout(wei)

        v = self.value(x) #(B,T,C)
        out = wei @ v #(B,T,T) @ (B,T,C) -> (B,T,C)
        return out
 
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x) #self attention
        x = self.ln1(x + y) #add and norm
        y = self.ffwd(x) #feed forward
        x = self.ln2(x + y) #add and norm
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights) #apply weight initialization with std = 0.02

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, index, max_new_tokens):
        #index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:] #crop context to block size
            #get predictions
            logits, loss = self.forward(index_cond)

            #focus only on last time step
            logits = logits[:, -1, :] #becomes(B,C)
            probs = F.softmax(logits, dim=-1) # (B,C)

            #sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B,1)

            #append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B,T+1)

        return index

model = GPTLanguageModel(vocab_size)
#print('loading model parameters from model-01.pkl...')
#with open('model-01.pkl', 'rb') as f:
#    model = pickle.load(f)
#print('model parameters loaded successfully.')
m = model.to(device)


#optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
    if iter % eval_iter == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")


    #sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to model-01.pkl")