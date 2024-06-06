import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
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
    model.eval()    # its cool to see when we have to use eval mode and train mode
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
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # register_buffer makes it so that tril is not a parameter of the model, but still occupies space.

    def forward(self, x):
        B,T,C = x.shape # batch_size, block_size, n_embed
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        # Compute attention scores ("Affinities")
        wei = q @ k.transpose(-2, -1)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        #perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1) # its like shifting from having one person right a big report to many persons writing a small report and concatinating those reports together. Reports are ofc about how much attention should one recieve
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
    
# Now we will create a class block. 
"""In the Self-Attention paper, We have a Block consisting of 
1. Masked MultiHead Attention
2. Multi-Head Attention, which is also using cross attention from another segment/part of the block
3. Feed Forward network
And this Block keeps on repeating.

In essence it does communication(1,2) followed by computation(3)
and it does so repeatedly
So, lets make a watered-down version of our own with 1 and 3."""

class Block(nn.Module):
    """ Transformer Block: Communication followed by Computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head # Head Size can be seen in megamind terms, like the quantity of data one head can return. n_head increases the quality by taking in several opinions. 
        # likewise, head_size improves quality of one heads opinion, using quantity as said above, but, n_head increases the quantity of heads thinking independently, with the results evaluated together in the end.
        self.sa = MultiHeadAttention(n_head, head_size) # Communication
        self.ffwd = FeedForward(n_embd) # Computation

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x
    
"""Just adding the Blocks infrastruct leads to poorer results than before. The reason could be -
We are starting to get to a really deep neural net, and deep neural nets suffer from optimization issues. (training loss is quite smaller than validation loss)

Now there are two optimization ideas implemented in the self-attention paper that drastically helped with the depth of the network and kept them optimizable - 
1. Residual networks - The side arrows to Add&Norm segements are skip connections/ residual networks. It is like we are transforming the input to output, but we also have a skip connection that directly leads to the output. This both are computed together via a + or similar operator. The reason this is useful is coz addition distributes weights equally, to both of its branched. So, now we have a gradient super-highway, that transfers gradient directly from o/p to i/p. The forked pathway(where comutation happens) are initialized in such a way, that they contribute very little to the o/p at the start. So, the computation is invisible at the start, and slowly starts to show as its contributing."""


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the token embeddings for logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #lets make another embedding table here, this time for positions
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # multihead attention

        # Now we are interspersing communication many many time(3)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

        

    def forward(self, idx, targets=None):

        B,T = idx.shape

        tok_embd = self.token_embedding_table(idx) # (B,T, embedding dimension or n_embd))
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) #(T,n_embd)   (basically arange works like range)

        embdings = tok_embd + pos_embd # (B,T,C)
        x = self.blocks(embdings)
        
        logits = self.lm_head(x) # (B,T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last block_size tokens
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

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))