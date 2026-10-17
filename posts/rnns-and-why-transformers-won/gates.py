import torch

# How much does the input at step 0 affect the state at step T? The size of
# d h_T / d x_0 for untrained cells of the same width, at initialization.
# The gates' biases decide how much a cell "starts out remembering": the
# LSTM's forget gate (PyTorch order: input, forget, cell, output) and the
# GRU's update gate (order: reset, update, new) keep the old state when near 1.
T, d = 50, 64

def gradient_back(cls, gate_bias=None):
    sizes = []
    for seed in range(5):
        torch.manual_seed(seed)
        cell = cls(d, d, batch_first=True)
        if gate_bias is not None:
            with torch.no_grad():
                cell.bias_ih_l0[d:2 * d].fill_(gate_bias); cell.bias_hh_l0[d:2 * d].fill_(0.0)
        x = torch.randn(1, T, d, requires_grad=True)
        out, _ = cell(x)
        out[0, -1].sum().backward()
        sizes.append([x.grad[0, t].norm().item() for t in (T - 1, T - 10, T - 25, 0)])
    return torch.tensor(sizes).median(0).values

for name, cls, bias in [("RNN", torch.nn.RNN, None),
                        ("GRU, default", torch.nn.GRU, None), ("GRU, update bias +1", torch.nn.GRU, 1.0),
                        ("LSTM, forget bias 0", torch.nn.LSTM, 0.0), ("LSTM, forget bias 1", torch.nn.LSTM, 1.0),
                        ("LSTM, forget bias 3", torch.nn.LSTM, 3.0)]:
    s = gradient_back(cls, bias)
    print(f"{name:20s}: gradient reaching the input 1 / 10 / 25 / 50 steps back: "
          + " / ".join(f"{v:.1e}" for v in s))