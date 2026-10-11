import time
import torch
from transformers import DynamicCache
from lm import model, tok, DEVICE

# Constrained decoding, and what jump-forward saves. The output must be
#   {"title": "<text>", "year": <4 digits>, "summary": "<text>"}
# Inside a string the model may pick any token without a quote, a backslash
# or a newline, or a token that closes the string: a quote, possibly followed
# by the start of the fixed text after it (such as '",'), since tokens don't
# respect the grammar's boundaries. A string is at most 40 tokens; the year
# takes digit tokens. Everything else is fixed text the model has no choice
# about.
#   plain:        every output token costs a forward pass, fixed text included;
#   jump-forward: a run of fixed text is appended at once and fed to the model
#                 in the next pass, as SGLang's compressed state machine does.
# Greedy decoding, five prompts. The two modes feed the same tokens, but in
# different groupings, so 16-bit rounding can flip a near-tie: the outputs
# are compared token by token.
SEGMENTS = [("text", '{"title": "'), ("str", 40), ("text", ', "year": '), ("digits", 4),
            ("text", ', "summary": "'), ("str", 40), ("text", "}")]
TITLES = ["Paged Attention", "The KV Cache", "Speculative Decoding", "Mixture of Experts", "Prompt Caching"]
sync = torch.mps.synchronize if DEVICE == "mps" else torch.cuda.synchronize if DEVICE == "cuda" else (lambda: None)

pieces = [tok.decode([i]) for i in range(len(tok))]
special = set(tok.all_special_ids) | {i for i in range(len(tok)) if i >= tok.vocab_size}
QUOTE = tok.encode('"')[0]
BODY = torch.tensor([i not in special and '"' not in p and "\\" not in p and "\n" not in p and p != ""
                     for i, p in enumerate(pieces)], device=DEVICE)

def closers(follow):
    """Tokens that close a string: a quote, then the start of the fixed text `follow`."""
    return torch.tensor([i not in special and p.startswith('"') and follow.startswith(p[1:])
                         for i, p in enumerate(pieces)], device=DEVICE)
DIGIT_OK = torch.tensor([p.isdigit() for p in pieces], device=DEVICE)
CLOSE = {n: closers(SEGMENTS[n + 1][1]) for n, (kind, _) in enumerate(SEGMENTS) if kind == "str"}

@torch.no_grad()
def generate(prompt, jump):
    cache, passes = DynamicCache(), 0
    def feed(ids):
        out = model(torch.tensor([ids], device=DEVICE), past_key_values=cache, use_cache=True)
        return out.logits[0, -1, :len(pieces)]
    logits, pending, out = feed(prompt), [], []                     # the prompt's prefill is not counted
    def choose(mask):
        nonlocal logits, pending, passes
        if logits is None:
            logits = feed(pending); passes += 1
        t = int(torch.where(mask, logits, -torch.inf).argmax())
        logits, pending = None, [t]
        out.append(t)
        return t
    skip = 0                                                        # fixed text a closing token already wrote
    for n, (kind, arg) in enumerate(SEGMENTS):
        if kind == "text":
            ids = tok.encode(arg[skip:]) if arg[skip:] else []
            skip = 0
            if jump:                                                # append the whole run, no pass
                pending = pending + ids if logits is None else ids
                logits = None
                out.extend(ids)
            else:
                for t in ids:                                       # one pass per fixed token
                    mask = torch.zeros(len(pieces), dtype=torch.bool, device=DEVICE); mask[t] = True
                    choose(mask)
        elif kind == "str":
            close = CLOSE[n]
            for k in range(arg):
                t = choose(BODY | close)
                if close[t]:
                    skip = len(pieces[t]) - 1
                    break
            else:
                mask = torch.zeros(len(pieces), dtype=torch.bool, device=DEVICE); mask[QUOTE] = True
                choose(mask)
        else:
            for k in range(arg):
                choose(DIGIT_OK)
    return out, passes

for jump in (False, True):
    total_tokens = total_passes = 0
    sync(); t0 = time.perf_counter()
    outs = []
    for title in TITLES:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": f"Describe the blog post titled '{title}' as JSON with its title, "
              "the year it was published (2026), and a short one-sentence summary."}],
            add_generation_prompt=True, return_dict=True)["input_ids"]
        out, passes = generate(prompt, jump)
        outs.append(out); total_tokens += len(out); total_passes += passes
    sync(); seconds = time.perf_counter() - t0
    if not jump:
        plain = outs
        print(repr(tok.decode(outs[0])))
    print(f"{'jump-forward' if jump else 'plain':12s}: {total_tokens} output tokens in {total_passes} forward passes, "
          f"{seconds:.1f} s")
    if jump:
        same = sum(a == b for a, b in zip(outs, plain))
        first = [next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None) for a, b in zip(outs, plain)]
        print(f"outputs identical to plain decoding: {same} of {len(outs)}; first differing token: {first}")
