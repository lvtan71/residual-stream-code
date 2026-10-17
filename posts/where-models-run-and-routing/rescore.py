import json
import re
import unicodedata
import numpy as np

# Rescoring route.py's answers (answers.json) and looking inside the cascade.
# route.py's own table used a matcher that dropped accented letters
# ("Brasília" failed) and accepted one capital per country; here accents are
# folded and the documented alternatives (La Paz for Bolivia, Nur-Sultan for
# Kazakhstan's capital 2019-2022, Bujumbura for Burundi's former capital) count.
exec(open("route.py").read().split("@torch.no_grad()")[0].split("PATHS = ")[0])   # imports only
src = open("route.py").read()
exec(src[src.index("rng = random.Random(0)"):src.index("@torch.no_grad()")])        # the same 150 items
res = json.load(open("answers.json"))
ALSO = {"Sucre": ["la paz"], "Astana": ["nur-sultan", "nursultan"], "Gitega": ["bujumbura"]}
fold = lambda t: re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode())
def right(text, truth):
    return any(fold(a) in fold(text) for a in [truth] + ALSO.get(truth, []))
S = np.array([right(r["text"], it[1]) for r, it in zip(res["small"], items)])
L = np.array([right(r["text"], it[1]) for r, it in zip(res["large"], items)])
conf = np.array([r["conf"] for r in res["small"]])
kind = np.array([it[2] for it in items])
print(f"{'':16s} {'small':>6s} {'large':>6s}")
for k in ("capital", "multiplication", "weekday"):
    m = kind == k
    print(f"{k:16s} {S[m].mean():6.0%} {L[m].mean():6.0%}   small's confidence when wrong: median {np.median(conf[m & ~S]):.2f}")
print(f"{'all':16s} {S.mean():6.0%} {L.mean():6.0%}\n")

rng = np.random.default_rng(0)
def report(name, esc):
    acc = np.where(esc, L, S)
    boot = [np.where(esc, L, S)[i].mean() - S[i].mean() for i in (rng.integers(0, len(S), len(S)) for _ in range(2000))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    by = ", ".join(f"{k} {int((esc & (kind == k)).sum())}" for k in ("capital", "multiplication", "weekday"))
    print(f"{name:40s} {acc.mean():5.0%}  gain over small {acc.mean() - S.mean():+5.0%} [{lo:+.0%}, {hi:+.0%}]  escalated: {by}")
for th in (0.7, 0.9):
    report(f"cascade, escalate below {th}", conf < th)
report("route by type: only capitals to small", kind != "capital")
report("route by type: only multiplication to large", kind == "multiplication")
report("always large", np.ones(len(S), bool))
report("oracle (escalate exactly when small is wrong)", ~S)
