import json
import math
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from gpt import GPT, Config

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Train the tiny GPT on the blog. Hyperparameters are chosen on a dev slice,
# the last 10% of the training text; the two held-out posts are scored once,
# at the end, with the choice already made.
full_train, held_out = open("blog_train.txt").read(), open("blog_val.txt").read()
cut = int(0.9 * len(full_train))
dev_split = {"fit": full_train[:cut], "score": full_train[cut:]}     # choose here
final_split = {"fit": full_train, "score": held_out}               # report here

def tokenizer_for(kind, text):
    if kind == "char":
        vocab = sorted(set(text))
        index = {c: i + 1 for i, c in enumerate(vocab)}                 # unseen characters share id 0
        return len(vocab) + 1, lambda s: torch.tensor([index.get(c, 0) for c in s]), vocab
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.train_from_iterator([text], trainers.BpeTrainer(
        vocab_size=2048, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    return bpe.get_vocab_size(), lambda s: torch.tensor(bpe.encode(s).ids), bpe

@torch.no_grad()
def bits_per_byte(model, data, text, T):
    """Every token scored once, each with at least T/2 tokens of context
    (after the first window): windows of T, stepped by T/2, scoring the
    second half of each."""
    model.eval()
    total, s = 0.0, 0
    while True:
        x = data[s:s + T + 1]
        if len(x) < 2:
            break
        logits = model(x[None, :-1])[0]
        first = 0 if s == 0 else T // 2
        total += F.cross_entropy(logits[first:], x[1 + first:], reduction="sum").item()
        if s + T + 1 >= len(data):
            break
        s += T // 2
    model.train()
    return total / math.log(2) / len(text.encode("utf-8"))

def run(kind, seed, split, steps, checkpoints=(), B=32, lr=1e-3, warmup=200, schedule="constant",
        device=DEVICE, on_checkpoint=None, track_fit=False, **cfg):
    V, encode, _ = tokenizer_for(kind, split["fit"])
    fit, score = encode(split["fit"]).to(device), encode(split["score"]).to(device)
    fit_n = len(encode(split["fit"][:60_000]))
    c = Config(V=V, **cfg)
    torch.manual_seed(seed)
    model = GPT(c).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]            # matrices, and stacks of expert matrices, decay; norm gains don't
    rest = [p for p in model.parameters() if p.dim() < 2]
    assert {id(p) for p in decay} | {id(p) for p in rest} == {id(p) for p in model.parameters()}   # every parameter in a group
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(seed)
    curve = {}
    for step in range(1, steps + 1):
        frac = min(1.0, step / warmup)
        if schedule == "cosine" and step > warmup:
            frac = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))
        for group in opt.param_groups:
            group["lr"] = lr * frac
        i = torch.randint(0, len(fit) - c.T - 1, (B,), generator=g).to(device)
        idx = i[:, None] + torch.arange(c.T + 1, device=device)
        x, y = fit[idx[:, :-1]], fit[idx[:, 1:]]
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        moes = [b.moe for b in model.blocks if b.moe is not None]
        if moes:                                                     # the balancing loss, if any ("Mixture of Experts")
            loss = loss + 0.01 * sum(m.aux_loss for m in moes)
            loads = [m.load for m in moes]
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in checkpoints or step == steps:
            curve[step] = bits_per_byte(model, score, split["score"], c.T)
            if track_fit:                                            # the same measure on 60k characters it trains on
                curve[f"{step} on training text"] = bits_per_byte(model, fit[:fit_n], split["fit"][:60_000], c.T)
            if on_checkpoint:
                on_checkpoint(step, model)
    passes = steps * B * c.T / len(fit)
    if moes:                                                         # the busiest expert's share, per layer, at the end
        curve["busiest expert"] = " ".join(f"{l.max().item():.0%}" for l in loads)
    return model, curve, passes

if __name__ == "__main__":
    # python train.py <name> '<json of run() keyword arguments>'
    name, kw = sys.argv[1], json.loads(sys.argv[2])
    if kw.pop("corpus", "blog") == "wikitext2":                # WikiText-2 in place of the blog (wikitext2.py writes it)
        full_train, held_out = open("wikitext2_train.txt").read(), open("wikitext2_test.txt").read()
        cut = int(0.9 * len(full_train))
        dev_split = {"fit": full_train[:cut], "score": full_train[cut:cut + 120_000]}   # 120k characters, to keep scoring fast
        final_split = {"fit": full_train, "score": held_out}
    split = final_split if kw.pop("final", False) else dev_split
    save = kw.pop("save", None)
    model, curve, passes = run(split=split, **kw)
    if save:
        cfg = {k: v for k, v in model.c.__dict__.items()}
        vocab = tokenizer_for(kw["kind"], split["fit"])[2] if kw["kind"] == "char" else None
        torch.save({"model": model.cpu().state_dict(), "config": cfg, "vocab": vocab}, save)
    print(f"{name}: " + " / ".join(f"{s}: {v:.3f}" if isinstance(v, float) else f"{s}: {v}"
                                    for s, v in curve.items()) + f"  ({passes:.0f} passes)", flush=True)