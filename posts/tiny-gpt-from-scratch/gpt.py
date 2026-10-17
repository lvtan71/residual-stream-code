import math
import torch
import torch.nn.functional as F

# A tiny GPT, every piece from the earlier posts: token embeddings ("Embeddings and Contrastive Learning"),
# RoPE ("Where Am I?"), pre-norm RMSNorm ("Dropout and Normalization"), causal multi-head attention
# ("The Transformer's Three Attention Patterns"), a SwiGLU MLP, residual connections ("The Residual Stream"), weight-tied
# unembedding, trained on next-token cross-entropy ("Cross-Entropy and the Softmax Gradient", "What 'Generative' Means").
class Config:
    def __init__(self, **kw):
        self.V, self.T, self.D, self.L, self.N = 2048, 256, 192, 4, 4     # vocab, context, width, layers, heads
        self.dropout, self.position, self.residual, self.prenorm = 0.1, "rope", True, True
        self.__dict__.update(kw)

def rope(x):                                          # x: (B, heads, T, H)
    T, H = x.shape[-2], x.shape[-1]
    ang = torch.arange(T, device=x.device)[:, None] * 10_000.0 ** (-torch.arange(0, H, 2, device=x.device) / H)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()], -1).flatten(-2)

class RMSNorm(torch.nn.Module):
    def __init__(self, D):
        super().__init__(); self.g = torch.nn.Parameter(torch.ones(D))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g

class Block(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c, F_ = c, int(8 * c.D / 3)              # SwiGLU hidden width: 8/3 D keeps the MLP at 8 D^2 weights
        self.n1, self.n2 = RMSNorm(c.D), RMSNorm(c.D)
        self.qkv, self.proj = torch.nn.Linear(c.D, 3 * c.D, bias=False), torch.nn.Linear(c.D, c.D, bias=False)
        self.w_gate, self.w_up = torch.nn.Linear(c.D, F_, bias=False), torch.nn.Linear(c.D, F_, bias=False)
        self.w_down = torch.nn.Linear(F_, c.D, bias=False)
        self.drop = torch.nn.Dropout(c.dropout)
    def attention(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.c.N, D // self.c.N).permute(2, 0, 3, 1, 4)
        if self.c.position == "rope":
            q, k = rope(q), rope(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.c.dropout if self.training else 0)
        return self.drop(self.proj(y.transpose(1, 2).reshape(B, T, D)))
    def mlp(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
    def forward(self, x):
        c = self.c
        if not c.prenorm:                              # the 2017 layout: add, then normalize
            x = self.n1(x + self.attention(x))
            return self.n2(x + self.mlp(x))
        if not c.residual:                             # no shortcut: each sublayer replaces the stream
            x = self.attention(self.n1(x))
            return self.mlp(self.n2(x))
        x = x + self.attention(self.n1(x))
        return x + self.mlp(self.n2(x))

class GPT(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embed = torch.nn.Embedding(c.V, c.D)
        self.pos = torch.nn.Embedding(c.T, c.D) if c.position == "learned" else None
        self.blocks = torch.nn.ModuleList([Block(c) for _ in range(c.L)])
        self.norm = RMSNorm(c.D)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, idx):
        x = self.embed(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.norm(x) @ self.embed.weight.T          # unembedding shares the embedding table
