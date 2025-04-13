import torch
from torch import nn
from torch.nn import functional as F


device = 'mps' if torch.mps.is_available() else 'cpu'
block_size = 256
emb_n = 364
n_heads = 8
n_layer = 8
dropout = 0.2

class Head(nn.Module):
    def __init__(self,head_size):
        super(Head,self).__init__()
        self.keys = nn.Linear(emb_n,head_size)
        self.queries = nn.Linear(emb_n,head_size)
        self.values = nn.Linear(emb_n,head_size)
        self.droput = nn.Dropout(0.2)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self,x):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        wei = k @ q.transpose(-1,-2) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0,float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.droput(wei)

        v = self.values(x)

        out = wei @ v
        return out
    

class MultiHead(nn.Module):
    def __init__(self,num_heads,head_size):
        super(MultiHead,self).__init__()
        self.layers = nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.projection = nn.Linear(num_heads*head_size,emb_n)
        self.droput = nn.Dropout(0.2)
    def forward(self,x):
        out = torch.concat([h(x) for h in self.layers], dim=-1)
        out = self.droput(self.projection(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Sequential(
            nn.Linear(emb_n,4*emb_n),
            nn.ReLU(),
            nn.Linear(4*emb_n, emb_n),
            nn.Dropout(0.2)
        )
        
    def forward(self,x):
        return self.l(x)


    
    

class Block(nn.Module):
    def __init__(self,n_embd,n_heads):
        super(Block,self).__init__()
        head_size = n_embd // n_heads
        self.multi_head = MultiHead(n_heads,head_size)
        self.ln1 = nn.LayerNorm(emb_n)
        self.ln2 = nn.LayerNorm(emb_n)
        self.fwd = FeedForward()
    def forward(self,x):
        x = x + self.multi_head(self.ln1(x))
        x = x + self.fwd(self.ln2(x))
        return x
    

class Model(nn.Module):
    def __init__(self,vocab_size,emb_n):
        super(Model,self).__init__()
        self.vec_space = nn.Embedding(vocab_size,emb_n)
        self.pos_space = nn.Embedding(block_size,emb_n)
        self.blocks = nn.Sequential(*[Block(emb_n,n_heads) for _ in range(n_layer)])
        self.l = nn.Linear(emb_n,vocab_size)
        self.ln = nn.LayerNorm(emb_n)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,x, targets=None):
        B, T = x.shape
        x = self.vec_space(x)
        x = x + self.pos_space(torch.arange(T).to(device))
        x = self.blocks(x)
        x = self.ln(x)
        x = self.l(x)

        if targets == None:
            loss = None
        else:
            B, T ,C = x.shape
            x = x.view(B*T,C)
            targets = targets.view(-1)
            loss = F.cross_entropy(x,targets)

        return x,loss
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx







