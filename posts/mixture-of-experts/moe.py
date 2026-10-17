import torch
import torch.nn.functional as F

# A mixture-of-experts MLP: E small SwiGLU experts and a router. Each token
# goes to its top-k experts, weighted by the router's softmax probabilities
# (renormalized over the chosen k). Balancing, one of:
#   "aux":  the Switch Transformer loss, E * sum_e f_e * P_e, where f_e is the
#           share of tokens routed to expert e and P_e its mean router
#           probability; minimized when both are uniform (Fedus et al., 2022);
#   "bias": DeepSeek-V3's auxiliary-loss-free method: a per-expert bias added
#           to the scores for *choosing* experts only, nudged after each step
#           toward under-used experts (Wang et al., 2024);
#   "none": nothing.
class MoE(torch.nn.Module):
    def __init__(self, D, F_, E, k, balance, bias_rate=1e-3, renorm_top1=False):
        super().__init__()
        self.E, self.k, self.balance, self.bias_rate = E, k, balance, bias_rate
        self.renorm_top1 = renorm_top1
        self.router = torch.nn.Linear(D, E, bias=False)
        self.gate = torch.nn.Parameter(torch.randn(E, D, F_) * 0.02)
        self.up = torch.nn.Parameter(torch.randn(E, D, F_) * 0.02)
        self.down = torch.nn.Parameter(torch.randn(E, F_, D) * 0.02)
        self.register_buffer("bias", torch.zeros(E))
        self.aux_loss, self.load = torch.tensor(0.0), torch.zeros(E)

    def forward(self, x):
        B, T, D = x.shape
        x = x.reshape(-1, D)
        probs = self.router(x).softmax(-1)                                # (tokens, E)
        choose_by = probs + self.bias if self.balance == "bias" else probs
        top = choose_by.topk(self.k, -1).indices                          # (tokens, k)
        weight = probs.gather(1, top)
        if self.k > 1 or self.renorm_top1:
            weight = weight / weight.sum(-1, keepdim=True)        # renormalize over the chosen k
        # With k = 1, renormalizing makes every weight exactly 1, so the
        # router would get no gradient from the language-model loss at all;
        # Switch Transformers uses the raw probability instead.
        out = torch.zeros_like(x)
        counts = torch.zeros(self.E, device=x.device)
        for e in range(self.E):                                           # run each expert on its tokens
            rows, slot = (top == e).nonzero(as_tuple=True)
            counts[e] = len(rows)
            if len(rows):
                h = x[rows]
                y = (F.silu(h @ self.gate[e]) * (h @ self.up[e])) @ self.down[e]
                out.index_add_(0, rows, y * weight[rows, slot, None])
        f = counts / counts.sum()
        self.load = f.detach()
        self.aux_loss = self.E * (f * probs.mean(0)).sum() if self.balance == "aux" else torch.tensor(0.0, device=x.device)
        if self.balance == "bias" and self.training:                      # raise the bias of under-loaded experts
            with torch.no_grad():
                self.bias += self.bias_rate * torch.sign(1 / self.E - f)
        return out.reshape(B, T, D)
