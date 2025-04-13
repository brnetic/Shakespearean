from model import Model
import torch
import pandas as pd

# Load the dataset



device = 'mps' if torch.mps.is_available() else 'cpu'


eval_iters = 10000
block_size = 256
emb_n = 384
batch_size = 64

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text),dtype=torch.long).to(device)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

model = Model(vocab_size,emb_n).to(device)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
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
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)

print(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

for iter in range(eval_iters):
    
    
    x,y = get_batch('train')

    logits, loss = model(x,y)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    if(iter == eval_iters//2):
        for paramaters in optimizer.param_groups:
            paramaters['lr'] = 0.5e-5

    if iter % 500 == 0:
        print(f"step {iter}: loss {loss}")
    
    

final_loss = estimate_loss()

print(f"Overall Train Loss: {final_loss['train']}, Overall Validation Loss: {final_loss['val']}")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
torch.save(model,"model.pth")
print(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))
    

    


    
