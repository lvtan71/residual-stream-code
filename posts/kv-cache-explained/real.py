import time
import torch
from transformers import AutoModelForCausalLM, DynamicCache
from lm import PATH, model, tok, DEVICE

# The KV cache in a real model, Qwen2.5-1.5B-Instruct: 28 layers, 2
# key-value heads of 128 dimensions (shared by its 12 query heads), 16-bit.
# 1. Exact: for 32 greedy steps, the next-token logits with the cache (one
#    new token) and by recomputing the whole sequence, in 32-bit, compared.
# 2. Memory: the cache's measured size after n tokens against the formula
#    2 x layers x kv heads x head dim x bytes x tokens.
# 3. Time: one generation step at context n, with the cache (one new token)
#    and without (the whole sequence again).
# 4. Batching: one decode step for B sequences of 512 tokens each.
# Prompts are slices of this blog. Time with nothing else using the GPU.
sync = torch.mps.synchronize if DEVICE == "mps" else torch.cuda.synchronize if DEVICE == "cuda" else (lambda: None)
blog = tok(open("blog_train.txt").read()).input_ids
cfg = model.config
L, KV, HD = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads

def timed(fn, trials=7):
    fn(); sync()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter(); fn(); sync()
        times.append(time.perf_counter() - t0)
    return sorted(times)[trials // 2]

with torch.no_grad():
    # 1. exact, in 32-bit so that rounding cannot hide a difference
    m32 = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32).to(DEVICE).eval()
    prompt = torch.tensor([blog[:256]], device=DEVICE)
    ids, worst, same = prompt, 0.0, 0
    cache = DynamicCache()
    last = m32(prompt, past_key_values=cache, use_cache=True).logits[0, -1]
    for step in range(32):
        full = m32(ids).logits[0, -1]                                   # the whole sequence again
        worst = max(worst, (full - last).abs().max().item())
        same += int(full.argmax() == last.argmax())
        nxt = full.argmax().view(1, 1)
        ids = torch.cat([ids, nxt], 1)
        last = m32(nxt, past_key_values=cache, use_cache=True).logits[0, -1]   # one new token
    print(f"1. 32 greedy steps after a 256-token prompt, 32-bit: same next token with and without the cache "
          f"at {same} of 32 steps; largest logit difference {worst:.1e} (logits span about "
          f"{full.max() - full.min():.0f})")
    del m32

    # 2. memory
    print("2. cache size     measured      formula 2 x L x KV x d x 2 bytes x n")
    for n in (512, 2048, 8192):
        cache = DynamicCache()
        model(torch.tensor([blog[:n]], device=DEVICE), past_key_values=cache, use_cache=True, logits_to_keep=1)
        measured = sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values))
        print(f"   {n:5,d} tokens  {measured / 2**20:8.1f} MiB   {2 * L * KV * HD * 2 * n / 2**20:8.1f} MiB "
              f"({2 * L * KV * HD * 2 / 1024:.0f} KiB per token)")

    # 3. one step, with and without the cache
    print("3. one generation step at context n:  with cache   without   ratio")
    for n in (256, 512, 1024, 2048, 4096):
        ids = torch.tensor([blog[:n + 1]], device=DEVICE)
        cache = DynamicCache()
        model(ids[:, :n], past_key_values=cache, use_cache=True, logits_to_keep=1)
        def step():                                                     # reuse the same n-token cache
            cache.crop(-1)
            model(ids[:, n:], past_key_values=cache, use_cache=True)
        with_cache = timed(step)
        without = timed(lambda: model(ids))
        print(f"   {n:5,d} tokens              {with_cache * 1e3:9.1f} ms  {without * 1e3:7.1f} ms  {without / with_cache:5.1f}x")

    # 4. batching: B sequences, 512 cached tokens each, one decode step
    print("4. batch   one decode step   tokens/s   x batch 1")
    base = None
    for B in (1, 4, 16, 64):
        seqs = torch.tensor([blog[i * 600:i * 600 + 513] for i in range(B)], device=DEVICE)
        cache = DynamicCache()
        model(seqs[:, :512], past_key_values=cache, use_cache=True, logits_to_keep=1)
        def step():
            cache.crop(-1)
            model(seqs[:, 512:], past_key_values=cache, use_cache=True)
        t = timed(step)
        base = base or B / t
        print(f"   {B:5d}   {t * 1e3:11.1f} ms   {B / t:8.0f}   {B / t / base:6.1f}x")
