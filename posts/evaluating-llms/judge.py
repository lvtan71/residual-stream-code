import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Qwen2.5-1.5B-Instruct as a pairwise judge, on questions whose right answer
# is known, so its biases can be measured rather than argued about. The
# judge sees a question and two answers and is asked which is correct; its
# choice is whichever of the tokens "A" and "B" it rates more likely next.
PATH = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                         revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float16).to(DEVICE).eval()
A, B = tok.encode("A")[0], tok.encode("B")[0]

@torch.no_grad()
def judge(question, first, second, labels=("A", "B")):
    """Returns 1 if the judge prefers the first-shown answer, else 2.
    labels: the letters printed before the first and second answers."""
    prompt = (f"Question: {question}\n\nAnswer {labels[0]}: {first}\n\nAnswer {labels[1]}: {second}\n\n"
              "Which answer is correct? Reply with a single letter, A or B.")
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to(DEVICE)
    logits = model(ids).logits[0, -1]
    first_letter = A if labels[0] == "A" else B
    second_letter = B if labels[0] == "A" else A
    return 1 if logits[first_letter] > logits[second_letter] else 2

rng = random.Random(0)
items = []                                   # (question, right answer, wrong answer)
for _ in range(100):
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    wrong = a * b + rng.choice([-1, 1]) * rng.randint(1, 9) * 10
    items.append((f"What is {a} x {b}?", str(a * b), str(wrong)))
capitals = [("Australia", "Canberra", "Sydney"), ("Canada", "Ottawa", "Toronto"), ("Brazil", "Brasilia", "Rio de Janeiro"),
            ("Turkey", "Ankara", "Istanbul"), ("Switzerland", "Bern", "Zurich"), ("Morocco", "Rabat", "Casablanca"),
            ("Nigeria", "Abuja", "Lagos"), ("Vietnam", "Hanoi", "Ho Chi Minh City"), ("New Zealand", "Wellington", "Auckland"),
            ("South Africa", "Pretoria", "Johannesburg"), ("Pakistan", "Islamabad", "Karachi"), ("Myanmar", "Naypyidaw", "Yangon"),
            ("Kazakhstan", "Astana", "Almaty"), ("Tanzania", "Dodoma", "Dar es Salaam"), ("Bolivia", "Sucre", "Santa Cruz"),
            ("Ecuador", "Quito", "Guayaquil"), ("India", "New Delhi", "Mumbai"), ("China", "Beijing", "Shanghai"),
            ("Italy", "Rome", "Milan"), ("Germany", "Berlin", "Munich")]
items += [(f"What is the capital of {c}?", r, w) for c, r, w in capitals]

def verbose(wrong):            # a wrong answer with claims of having been checked
    return (f"The answer is {wrong}. I worked this out carefully, checking each step twice, and the result is "
            f"consistent with standard references, so you can rely on {wrong} being correct.")
def padded(ans):               # the same length of padding, with no claims
    return (f"The answer is {ans}. This question asks for one value, and the reply gives that value here in "
            f"a full sentence rather than on its own, followed by this sentence, which adds nothing to {ans}.")

def rate(pairs, labels=("A", "B")):
    """Share of judgments that pick the right answer, over both orders.
    pairs: (question, answer that is right, the other answer)."""
    hits = [judge(q, r, w, labels) == 1 for q, r, w in pairs] + [judge(q, w, r, labels) == 2 for q, r, w in pairs]
    return sum(hits) / len(hits)

subsets = {"multiplication": items[:100], "capitals": items[100:]}
print(f"{len(items)} questions with a known answer: 100 two-digit multiplications, {len(capitals)} capitals\n")
print(f"{'':40s} {'multiplication':>15s} {'capitals':>9s}")
def row(name, f):
    print(f"{name:40s} " + " ".join(f"{f(v):>{w}.0%}" for v, w in zip(subsets.values(), (15, 9))), flush=True)
row("right answer first: judge right", lambda p: sum(judge(q, r, w) == 1 for q, r, w in p) / len(p))
row("right answer second: judge right", lambda p: sum(judge(q, w, r) == 2 for q, r, w in p) / len(p))
row("  same, first answer labeled B", lambda p: sum(judge(q, w, r, ("B", "A")) == 2 for q, r, w in p) / len(p))
row("right in both orders", lambda p: sum(judge(q, r, w) == 1 and judge(q, w, r) == 2 for q, r, w in p) / len(p))
row("chose the same slot both times", lambda p: sum(judge(q, r, w) == judge(q, w, r) for q, r, w in p) / len(p))
print("\nshare of judgments picking the right answer, both orders averaged:")
row("bare right vs bare wrong", lambda p: rate(p))
row("bare right vs wrong + claims of checking", lambda p: rate([(q, r, verbose(w)) for q, r, w in p]))
row("bare right vs wrong + neutral padding", lambda p: rate([(q, r, padded(w)) for q, r, w in p]))
row("right + neutral padding vs bare wrong", lambda p: rate([(q, padded(r), w) for q, r, w in p]))