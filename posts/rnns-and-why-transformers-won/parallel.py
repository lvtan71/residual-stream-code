import time
import torch

# The same 1,024-token sequence at width 256: a GRU must take 1,024 steps one
# after another; self-attention does it as a few big matrix multiplies.
torch.manual_seed(0)
T, d = 1024, 256
x = torch.randn(1, T, d)
gru = torch.nn.GRU(d, d, batch_first=True)
Wq, Wk, Wv = (torch.randn(d, d) / d ** 0.5 for _ in range(3))

def attention(x):
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    mask = torch.ones(T, T).triu(1).bool()                 # each token sees only the past
    s = (q @ k.transpose(-1, -2) / d ** 0.5).masked_fill(mask, float("-inf"))
    return torch.softmax(s, -1) @ v

def timed(fn, trials=10):
    fn(); times = []
    for _ in range(trials):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    return sorted(times)[trials // 2]

with torch.no_grad():
    t_rnn, t_att = timed(lambda: gru(x)), timed(lambda: attention(x))
flops_rnn = T * 3 * 2 * 2 * d * d                           # 3 gates, input and hidden matmuls
flops_att = 3 * 2 * T * d * d + 2 * 2 * T * T * d           # projections, then scores and weighted sum
print(f"GRU:       {T} sequential steps, {flops_rnn / 1e9:.2f} GFLOPs, {t_rnn * 1e3:6.1f} ms")
print(f"attention: 1 parallel pass,      {flops_att / 1e9:.2f} GFLOPs, {t_att * 1e3:6.1f} ms")