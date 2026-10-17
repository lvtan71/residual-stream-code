import math
import os
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# What a computer-use agent's observations cost. Screenshots: Claude's
# documented rule, one visual token per 28x28 patch, after shrinking the
# image (keeping its shape) to fit a tier's long-edge and token limits.
TIERS = {"standard": (1568, 1568), "high-resolution": (2576, 4784)}     # (max long edge, max visual tokens)
def visual_tokens(w, h, tier):
    edge, cap = TIERS[tier]
    tokens = lambda w, h: math.ceil(w / 28) * math.ceil(h / 28)
    if max(w, h) <= edge and tokens(w, h) <= cap:
        return w, h, tokens(w, h)
    lo, hi = 0.0, min(1.0, edge / max(w, h))            # largest scale that fits both limits
    for _ in range(60):
        mid = (lo + hi) / 2
        (lo, hi) = (mid, hi) if tokens(round(w * mid), round(h * mid)) <= cap else (lo, mid)
    return round(w * lo), round(h * lo), tokens(round(w * lo), round(h * lo))

print("screenshot tokens (check against the documentation's table: 1920x1080 -> 1456x819, 1560 / 2691)")
for w, h in ((1024, 768), (1280, 800), (1920, 1080), (2560, 1440)):
    row = [f"{w}x{h}"]
    for tier in TIERS:
        sw, sh, t = visual_tokens(w, h, tier)
        row.append(f"{tier}: {t:5,d} tokens" + ("" if (sw, sh) == (w, h) else f" (shrunk to {sw}x{sh})"))
    print("  " + "   ".join(row))

# Accessibility trees: the same pages as text, captured with Chrome's
# devtools at 1280x800 and counted with Qwen2.5's tokenizer (an approximate
# stand-in for any model's). A raw snapshot covers the whole page, with every
# node, URL and identifier; agents usually get a pruned version, here only
# the interactive elements with their labels, URLs dropped.
tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",       # the tokenizer only
                                                      revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                                                      allow_patterns=["*.json", "merges.txt"]))
import re
INTERACTIVE = ("link", "button", "textbox", "combobox", "checkbox", "radio", "searchbox", "menuitem", "tab", "option")
def pruned(text):
    keep = [l for l in text.splitlines() if l.strip().split(" ")[1:2] and l.strip().split(" ")[1] in INTERACTIVE]
    return "\n".join(re.sub(r' url="[^"]*"', "", l.strip()) for l in keep)
print("\naccessibility trees, whole page (raw devtools snapshot / interactive elements only, no URLs):")
for name in ("blog_home", "hackernews", "wikipedia"):
    if not os.path.exists(f"a11y_{name}.txt"):         # snapshots of live pages that were not kept (see the README)
        continue
    text = open(f"a11y_{name}.txt").read()
    p = pruned(text)
    print(f"  {name:11s} {len(text.splitlines()):6,d} nodes {len(tok(text).input_ids):9,d} tokens   "
          f"pruned: {len(p.splitlines()):5,d} elements {len(tok(p).input_ids):8,d} tokens")

# A task's input tokens. Each step the agent sees the newest screenshot
# plus the conversation so far; history grows unless old screenshots are
# dropped. 2,000 tokens of instructions and tools, 150 tokens of the
# model's own reasoning and action per step; output tokens not counted.
# With prompt caching ("Prompt Caching") a request reads the previous request's
# prefix at 0.1x the price and writes what's new at 1.25x. Dropping old
# screenshots changes the prefix, so a trimmed history can't be cached.
def task_tokens(steps, shot, keep=None):
    total = 0
    for t in range(1, steps + 1):
        shots = t if keep is None else min(t, keep)
        total += 2000 + 150 * t + shots * shot
    return total
def cached_cost(steps, shot, price=5e-6):
    cost, prev = 0.0, 0
    for t in range(1, steps + 1):
        now = 2000 + (150 + shot) * t
        cost += (0.1 * prev + 1.25 * (now - prev)) * price
        prev = now
    return cost
shot = visual_tokens(1280, 800, "standard")[2]
print(f"\none task's input tokens, screenshots at 1280x800 ({shot:,} tokens each), $5 per million input tokens:")
for steps in (10, 30, 50):
    a, b = task_tokens(steps, shot), task_tokens(steps, shot, keep=3)
    print(f"  {steps:3d} steps: keep every screenshot {a:>9,d} tokens (${a * 5e-6:5.2f}, cached ${cached_cost(steps, shot):5.2f}); "
          f"keep the last 3 {b:>8,d} (${b * 5e-6:5.2f}); last request {2000 + (150 + shot) * steps:,} tokens")
