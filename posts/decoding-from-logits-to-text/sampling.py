import torch

# Decoding policies, each a function from the model's logits for the next
# token to a choice. They're written as edits to the logits (masking a token
# means setting its logit to -inf), applied in a fixed order, then one draw.

def repetition_penalty(logits, history, theta):
    """Repetition penalty (Keskar et al., 2019): every token already generated
    becomes less likely. CTRL divides its logit by theta; dividing a negative
    logit would raise it, so (as in Hugging Face) negative ones are multiplied."""
    if theta == 1.0 or not history:
        return logits
    seen = torch.tensor(sorted(set(history)), device=logits.device)
    l = logits[seen]
    logits[seen] = torch.where(l > 0, l / theta, l * theta)
    return logits

def frequency_presence(logits, history, frequency, presence):
    """The OpenAI-style penalties: subtract frequency * (times the token has
    appeared) and presence * (1 if it has appeared at all)."""
    if not history or (frequency == 0 and presence == 0):
        return logits
    counts = torch.bincount(torch.tensor(history, device=logits.device), minlength=logits.numel()).float()
    return logits - frequency * counts - presence * (counts > 0).float()

def top_k(logits, k):
    """Keep the k most likely tokens."""
    if k is None:
        return logits
    kth = logits.topk(k).values[-1]
    return logits.masked_fill(logits < kth, float("-inf"))

def top_p(logits, p):
    """Nucleus sampling (Holtzman et al., 2020): keep the smallest set of the
    most likely tokens whose probabilities add up to at least p."""
    if p is None:
        return logits
    sorted_logits, order = logits.sort(descending=True)
    cumulative = sorted_logits.softmax(-1).cumsum(-1)
    drop = cumulative - sorted_logits.softmax(-1) >= p           # the mass *before* this token already reached p
    return logits.masked_fill(drop.scatter(0, order, drop), float("-inf"))

def min_p(logits, p_base):
    """Min-p (Nguyen et al., 2025): keep tokens with probability at least
    p_base times the top token's probability."""
    if p_base is None:
        return logits
    probs = logits.softmax(-1)
    return logits.masked_fill(probs < p_base * probs.max(), float("-inf"))

def choose(logits, history, generator, temperature=1.0, k=None, p=None, mp=None,
           rep=1.0, freq=0.0, pres=0.0):
    """One decoding step: penalties, then temperature, then the filters, then
    a draw. Temperature 0 means greedy: always the most likely token."""
    logits = logits.float().clone()
    logits = repetition_penalty(logits, history, rep)
    logits = frequency_presence(logits, history, freq, pres)
    if temperature == 0:
        return logits.argmax().item()
    logits = logits / temperature
    logits = min_p(top_p(top_k(logits, k), p), mp)
    return torch.multinomial(logits.softmax(-1), 1, generator=generator).item()

@torch.no_grad()
def generate(model, prompt_ids, n, seed=0, stop=(), **policy):
    """Generate up to n tokens with a KV cache: the prompt once, then one
    token per step ("What Generative Means" and "The KV Cache"). Stops early at any token in `stop`."""
    g = torch.Generator(device=prompt_ids.device).manual_seed(seed)
    out = model(prompt_ids[None], use_cache=True)
    history = []
    for _ in range(n):
        token = choose(out.logits[0, -1], history, g, **policy)
        history.append(token)
        if token in stop:
            break
        out = model(torch.tensor([[token]], device=prompt_ids.device),
                    past_key_values=out.past_key_values, use_cache=True)
    return history
