import json

# The error analysis of the greedy chained steps that serial_steps.py logs to
# chain_steps.json: how many steps went wrong, how many of those were
# multiplications or started from a number of 1,000 or more, the accuracy of
# steps starting below 1,000, how many wrong steps had a negative input or
# subtracted past zero, and the per-step accuracy on the eight-step problems
# (the last 320 steps: 40 problems of 8 steps).
S = json.load(open("chain_steps.json"))
W = [s for s in S if not s["right"]]
print(len(S), len(W), "wrong; multiply", sum(s["op"].startswith("multiply") for s in W),
      "; from >=1000", sum(abs(s["x"]) >= 1000 for s in W))
small = [s for s in S if abs(s["x"]) < 1000]
print("below 1000 right", sum(s["right"] for s in small) / len(small))
print("negatives or crossing zero among wrong:",
      sum(s["x"] < 0 or (s["op"].startswith("subtract") and int(s["op"].split()[-1]) > s["x"]) for s in W))
print("last 320 (k=8):", sum(s["right"] for s in S[-320:]) / 320)
