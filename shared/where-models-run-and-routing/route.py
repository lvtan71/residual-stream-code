import json
import random
import re
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # notebook: the post ran on "mps"
gpu_sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(GPU, lambda: None)
gpu_empty_cache = {"cuda": torch.cuda.empty_cache, "mps": torch.mps.empty_cache}.get(GPU, lambda: None)

# Routing between a small and a large model. Small: Qwen2.5-1.5B-Instruct;
# large: Qwen2.5-3B-Instruct (stand-ins for an on-device model and a bigger
# served one). 150 questions with checkable answers, of three kinds: capital
# cities, two-digit multiplications, and day-of-week arithmetic. Each model
# answers every question greedily; the small model's confidence is the
# probability it assigned to its own answer's tokens. A cascade sends a
# question to the large model when the small one's confidence is below a
# threshold. Cost: each model's measured time per question on this machine.
PATHS = {"small": snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                                    revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
         "large": snapshot_download("Qwen/Qwen2.5-3B-Instruct",
                                    revision="aa8e72537993ba99e69dfaafa59ed015b17504d1")}
rng = random.Random(0)
caps = [("France", "Paris"), ("Japan", "Tokyo"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Brazil", "Brasilia"),
        ("Turkey", "Ankara"), ("Switzerland", "Bern"), ("Nigeria", "Abuja"), ("Vietnam", "Hanoi"), ("Kenya", "Nairobi"),
        ("Morocco", "Rabat"), ("Pakistan", "Islamabad"), ("Myanmar", "Naypyidaw"), ("Kazakhstan", "Astana"), ("Peru", "Lima"),
        ("Egypt", "Cairo"), ("Norway", "Oslo"), ("Chile", "Santiago"), ("Tanzania", "Dodoma"), ("Bolivia", "Sucre"),
        ("Sri Lanka", "Colombo"), ("Ivory Coast", "Yamoussoukro"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
        ("Poland", "Warsaw"), ("Ethiopia", "Addis Ababa"), ("Ghana", "Accra"), ("Mongolia", "Ulaanbaatar"), ("Laos", "Vientiane"),
        ("Benin", "Porto-Novo"), ("Belize", "Belmopan"), ("Bhutan", "Thimphu"), ("Malawi", "Lilongwe"), ("Burundi", "Gitega"),
        ("Palau", "Ngerulmud"), ("Eswatini", "Mbabane"), ("Iceland", "Reykjavik"), ("Cuba", "Havana"), ("Nepal", "Kathmandu"),
        ("Uganda", "Kampala"), ("Zambia", "Lusaka"), ("Oman", "Muscat"), ("Qatar", "Doha"), ("Jordan", "Amman"),
        ("Latvia", "Riga"), ("Estonia", "Tallinn"), ("Slovenia", "Ljubljana"), ("Malta", "Valletta"), ("Fiji", "Suva")]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
items = [(f"What is the capital of {c}? Answer with just the city name.", a, "capital") for c, a in caps]
for _ in range(50):
    a, b = rng.randint(11, 99), rng.randint(11, 99)
    items.append((f"What is {a} times {b}? Answer with just the number.", str(a * b), "multiplication"))
for _ in range(50):
    d, n = rng.randrange(7), rng.randint(2, 60)
    items.append((f"Today is {DAYS[d]}. What day of the week will it be in {n} days? Answer with just the day.",
                  DAYS[(d + n) % 7], "weekday"))

@torch.no_grad()
def answer_all(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(GPU).eval()
    out = []
    for q, truth, kind in items:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)["input_ids"].to(GPU)
        gpu_sync(); t0 = time.perf_counter()
        g = model.generate(ids, max_new_tokens=12, do_sample=False, temperature=None, top_p=None, top_k=None,
                           output_scores=True, return_dict_in_generate=True)
        gpu_sync(); dt = time.perf_counter() - t0
        new = g.sequences[0, ids.shape[1]:]
        logp = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(g.scores, new)]
        text = tok.decode(new, skip_special_tokens=True).strip()
        right = re.sub(r"[^a-z0-9]", "", truth.lower()) in re.sub(r"[^a-z0-9]", "", text.lower())
        out.append({"right": right, "conf": float(np.exp(np.mean(logp[:max(1, len(logp) - 1)]))), "time": dt, "text": text})
    del model; gpu_empty_cache()
    return out

res = {name: answer_all(p) for name, p in PATHS.items()}
json.dump(res, open("answers.json", "w"))
S, L = res["small"], res["large"]
ts, tl = np.mean([r["time"] for r in S]), np.mean([r["time"] for r in L])
print(f"{len(items)} questions; seconds per question: small {ts:.2f}, large {tl:.2f}\n")
print(f"{'':10s} {'small':>7s} {'large':>7s}   (accuracy by kind)")
for kind in ("capital", "multiplication", "weekday"):
    idx = [i for i, it in enumerate(items) if it[2] == kind]
    print(f"{kind:14s} {np.mean([S[i]['right'] for i in idx]):5.0%} {np.mean([L[i]['right'] for i in idx]):7.0%}")
print(f"{'all':14s} {np.mean([r['right'] for r in S]):5.0%} {np.mean([r['right'] for r in L]):7.0%}\n")

# The cascade: every question goes to the small model first; below the
# confidence threshold it is also sent to the large model, whose answer is used.
print(f"{'policy':34s} {'accuracy':>9s} {'sent to large':>14s} {'seconds/question':>17s}")
def report(name, use_large):
    acc = np.mean([L[i]["right"] if u else S[i]["right"] for i, u in enumerate(use_large)])
    cost = ts + np.mean(use_large) * tl if "cascade" in name or "oracle" in name else (tl if all(use_large) else ts)
    print(f"{name:34s} {acc:9.0%} {np.mean(use_large):14.0%} {cost:17.2f}")
report("always small", [False] * len(items))
report("always large", [True] * len(items))
for th in (0.5, 0.7, 0.8, 0.9, 0.95):
    report(f"cascade, escalate below {th:.2f}", [r["conf"] < th for r in S])
report("oracle cascade (escalate if wrong)", [not r["right"] for r in S])
right = np.array([r["right"] for r in S]); conf = np.array([r["conf"] for r in S])
print(f"\nsmall model: confidence of right answers median {np.median(conf[right]):.2f}, "
      f"of wrong answers {np.median(conf[~right]):.2f}")
