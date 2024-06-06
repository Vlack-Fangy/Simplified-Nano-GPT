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

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the token embeddings for logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # previously, we just recieved an embedding of vocab size, through which we directly deduced the Y. But this way, we come to an intermediate stage, where embeddings represent a level of understanding for the input data.

        #lets make another embedding table here, this time for positions
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # we are creating a Head object in here
        # self.sa_head = Head(n_embd)  # head_size = n_embed

        # multihead attention
        self.sa_heads = MultiHeadAttention(4, n_embd//4)
        # multi headed attention is better than single head attention coz, it helps us gather a lots of different types of perspective(different types of data, like one could be the significance of vovels before, another for consonants, etc...) to solve one problem.

        self.ffwd = FeedForward(n_embd)

        # and for converting token embedding to logits, we will need a linear layer
        # ground work for using ermbeddings or projecting these values in a meaningful vactor space for better interaction and capabilities for understanging positional embeddings
        self.lm_head = nn.Linear(n_embd, vocab_size)

        

    def forward(self, idx, targets=None):

        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embedding_table(idx) # (B,T, embedding dimension or n_embd))
        # unlike previously, we dont want to go through embedding for the logits but go through an intermediate phase there

        # also, so far we have used this indexes to get logits and not as part of anything, sparing the importance that should be given to spatial arrangements.
        # now we wont just be encoding the identity of these tokens, but also their positions.
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) #(T,n_embd)   (basically arange works like range)

        embdings = tok_embd + pos_embd # (B,T,C)
        # pos_embd get right aligned, a new dimension of 1 gets added, and it gets broadcasted across batch
        # currently its not very useful as we are working on Bigram model, but later this will prove useful

        #feeding these values to self-attention head
        x = self.sa_heads(embdings)

        x = self.ffwd(x)
        # this feedforward layer is important coz, while we are having encodings that represent the input and also consider attention and positions, we want a bit of computation to think about it and work using its patterns and everything, instead of directly jumping to the answers

        # now lets get the logits-
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
            idx_cond = idx[:, -block_size:] # as if its not here, we wont be able to use position embedding table well, as it will give index out of range. Positional_embed_table has embeddings only upto bloack_size.

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