import torch
import torch.nn as nn
from torch.nn import functional as F

#
batch_size = 64
block_size = 256
max_iters=5000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

#--------------

torch.manual_seed(1337)


# 1. 读取数据
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi= { ch:i for i,ch in enumerate(chars) } 
itos= { i:ch for i,ch in enumerate(chars) } 
encode= lambda s: [stoi[c] for c in s]
decode= lambda l: ''.join([itos[i] for i in l])

# 2. 数据集划分
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


#数据加载
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
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

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # 注意这里应该是 dim=1，因为我们要对每一行进行 softmax
        wei = self.dropout(wei)

        v=self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out =torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),    
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
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
        x = x+ self.sa(self.ln1(x)) 
        x = x+ self.ffwd(self.ln2(x)) 
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个嵌入表 (Embedding Table)
        # 行数 = vocab_size (每个可能的输入字符对应一行)
        # 列数 = n_embd (每个可能的输出字符对应一列，代表 logits)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx 和 targets 都是 (B,T) 的整数张量
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb= self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C) 位置编码和 token 编码相加
        x = self.blocks(x) # (B,T,C)
        #x = self.ffwd(x) #(B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # --- 修复代码开始 ---
            B, T, C = logits.shape
            # 1. 重新变形 logits，变成 (Batch*Time, Channel) 即 (32, 65)
            logits = logits.view(B*T, C)
            # 2. 重新变形 targets，变成 (Batch*Time) 即 (32)
            targets = targets.view(B*T)
            # 3. 现在形状符合标准 (N, C) 和 (N) 了
            loss = F.cross_entropy(logits, targets)
            # --- 修复代码结束 ---
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: 这里的输入 idx 就是我们常说的 "Prompt"（提示词）。
        # 它的形状是 (B, T)，比如 (1, 1) 代表我们给模型一个初始字符（例如回车符 '0'）。
        
        for _ in range(max_new_tokens): # 循环 100 次，生成 100 个新字符
            
            idx_cond = idx[:, -block_size:] # 保持输入长度不超过 block_size
            # 1. 预测 (Get Predictions)
            # 把当前的整个序列 idx 扔给模型
            logits, loss = self(idx_cond) # (B, T, C)，这里的 T 是 idx_cond 的长度
            
            # 2. 只看最后一步 (Focus on last time step)
            # 模型虽然输出了对 idx 里每一个字符的预测，但我们只关心最后一个！
            # 因为我们要预测的是“未来”的那个新字符。
            logits = logits[:, -1, :] # 形状变成 (B, C)
            
            # 3. 计算概率 (Softmax)
            # 把得分变成概率分布。比如 ['a': 0.1, 'b': 0.8, 'c': 0.1]
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # 4. 随机抽样 (Sample)
            # 依据概率随机选这一个字。
            # 这里用 multinomial 而不是 argmax（直接选概率最大的），是为了保证生成的文本具有多样性。
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # 5. 拼接 (Concatenate)
            # 把新预测出来的字符 idx_next 拼到原来的 idx 后面。
            # 此时 idx 长度 +1，并在下一次循环中作为输入。
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx
   

model  = BigramLanguageModel()
m = model.to(device)


#建立优化器
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # 每 eval_interval 评估一次损失
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # 获取一个小批量数据
    xb, yb = get_batch('train')

    # 前向传播，计算损失
    logits, loss = model(xb, yb)
    

    # 后向传播，更新参数
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# 生成文本
context = torch.zeros((1, 1), dtype=torch.long, device=device) # 从一个回车符开始
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))