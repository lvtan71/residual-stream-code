import collections
import itertools
import json
import math
import re
import random
import sys

# Why each wrong answer was wrong. Rebuilds the same companies and questions
# as arch.py (same seed), then sorts every logged trial into: right; the
# sign flipped; a total missing one company; no answer; other.
rng = random.Random(1)
names = ["Arvelo", "Brindle", "Cobaltic", "Dunmore", "Elstree", "Farrow", "Gilden", "Harrowgate", "Istria",
         "Juniper Labs", "Kestrel", "Lumen Forge"]
cities = ["Lyon", "Osaka", "Porto", "Tartu", "Quito", "Perth"]
db = {n: {"founded": rng.randint(1950, 2020), "employees": rng.randint(40, 9000), "city": rng.choice(cities)}
      for n in names}
kinds, parts = [], []
for _ in range(20):
    a, b, c = rng.sample(names, 3)
    kind = rng.randrange(3)
    kinds.append(["years older", "total of three", "more employees"][kind])
    parts.append([db[x]["employees"] for x in (a, b, c)] if kind == 1 else [])

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k / n:4.0%} [{c - h:.0%}, {c + h:.0%}]"

# The log arch.py wrote; pass arch_trials_m4.jsonl for the post's own trials.
log = sys.argv[1] if len(sys.argv) > 1 else "arch_trials_v3.jsonl"
trials = [json.loads(l) for l in open(log)]
# The order the post prints them in, whatever order the log has them.
archs = ["ReAct", "reflection", "plan-and-execute"]
print(f"{'':18s} {'right [95% CI]':>18s} {'sign flipped':>13s} {'total missing one':>18s} {'no answer':>10s} {'other':>6s}")
for a in archs:
    c = collections.Counter()
    ts = [t for t in trials if t["arch"] == a]
    for t in ts:
        p, y, i = t["pred"], t["truth"], t["i"]
        if p == y: c["right"] += 1
        elif p is None: c["none"] += 1
        elif p == -y: c["sign"] += 1
        elif parts[i] and p in {x + z for x, z in itertools.combinations(parts[i], 2)}: c["missing"] += 1
        else: c["other"] += 1
    print(f"{a:18s} {wilson(c['right'], len(ts)):>18s} {c['sign']:>13d} {c['missing']:>18d} {c['none']:>10d} {c['other']:>6d}")
print("\nright, by question type:")
for a in archs:
    ts = [t for t in trials if t["arch"] == a]
    row = []
    for k in ("years older", "more employees", "total of three"):
        sub = [t for t in ts if kinds[t["i"]] == k]
        row.append(f"{k} {sum(t['pred'] == t['truth'] for t in sub)}/{len(sub)}")
    print(f"  {a:18s} " + ", ".join(row))

# Paired: same question, same run index, two architectures.
print("\npaired (same question and run): only the first right / only the second right")
ok = {(t["arch"], t["i"], t["k"]): t["pred"] == t["truth"] for t in trials}
for a, b in itertools.combinations(archs, 2):
    keys = [(i, k) for (x, i, k) in ok if x == a and (b, i, k) in ok]
    print(f"  {a} vs {b}: {sum(ok[(a, *q)] and not ok[(b, *q)] for q in keys)} / "
          f"{sum(ok[(b, *q)] and not ok[(a, *q)] for q in keys)}")

# Reflection: its first attempt is ReAct's run (same seed, same prompt),
# and the critic's reply is logged, so each trial shows what it did.
r = {(t["i"], t["k"]): t for t in trials if t["arch"] == "ReAct"}
f = [t for t in trials if t["arch"] == "reflection"]
flagged = [t for t in f if not re.match(r"\W*OK\b", t["critique"].strip().upper())]
first_ok = lambda t: r[(t["i"], t["k"])]["pred"] == t["truth"]
c = collections.Counter((first_ok(t), t["pred"] == t["truth"]) for t in flagged)
print(f"\nreflection: the critic flagged {len(flagged)} of {len(f)} first attempts "
      f"({sum(map(first_ok, flagged))} of the {sum(map(first_ok, f))} right ones, "
      f"{len(flagged) - sum(map(first_ok, flagged))} of the {len(f) - sum(map(first_ok, f))} wrong ones)")
print(f"  after the retry: wrong->right {c[(False, True)]}, right->wrong {c[(True, False)]}, "
      f"wrong->wrong {c[(False, False)]}, right->right {c[(True, True)]}")
signs = [t for t in f if r[(t["i"], t["k"])]["pred"] == -t["truth"]]
print(f"  first attempts with the sign flipped: {len(signs)}; flagged {sum(t in flagged for t in signs)}; "
      f"right after the retry {sum(t['pred'] == t['truth'] for t in signs)}")

# The counts the post quotes from the replies themselves.
yo = [t for t in trials if kinds[t["i"]] == "years older" and t["pred"] != t["truth"]]
print(f"\nyears older, wrong: {len(yo)}; the right number with the wrong sign {sum(t['pred'] == -t['truth'] for t in yo)}")
rt = [t for t in trials if t["arch"] == "ReAct" and kinds[t["i"]] == "total of three" and t["pred"] != t["truth"]]
said = [t for t in rt if re.search(r"issue|couldn't", t["reply"])]
print(f"ReAct, wrong totals: {len(rt)}; the reply says a lookup failed {len(said)}")
praised = [t for t in flagged if "correct" in re.split(r"[.:]", t["critique"])[0].lower()]
print(f"reflection: critic replies other than OK {len(flagged)}; first sentence calls it correct {len(praised)}")
none = [t for t in f if t["pred"] is None]
print(f"reflection: unanswered {len(none)}; after a retry {sum(t in flagged for t in none)}")
