import math
from statistics import NormalDist

# How many questions does an eval need? Two models, accuracy around 70%,
# and a difference of 2 points to detect, with a 5% false-alarm rate
# (two-sided) and 80% power.
z = NormalDist().inv_cdf
za, zb = z(0.975), z(0.80)
p1, p2 = 0.70, 0.72
# Unpaired: the two models answer different random questions.
n_unpaired = (za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
print(f"unpaired (different questions): {math.ceil(n_unpaired):,} questions per model")
# Paired: both answer the same questions, and only the questions they
# disagree on carry information (McNemar). If they disagree on a share d of
# questions, the per-question difference has variance d - (p2 - p1)^2,
# about d. (Independent errors would make them disagree on
# p1 (1 - p2) + p2 (1 - p1) = 41%: the unpaired case.)
for d in (0.30, 0.15, 0.08):
    n = (za + zb) ** 2 * d / (p2 - p1) ** 2
    print(f"paired, models disagree on {d:.0%} of questions: {math.ceil(n):,} questions")

# A 95% confidence interval for one accuracy (Wilson score interval).
def wilson(k, n, conf=0.95):
    q = z(0.5 + conf / 2)
    p = k / n
    centre = (p + q * q / (2 * n)) / (1 + q * q / n)
    half = q * math.sqrt(p * (1 - p) / n + q * q / (4 * n * n)) / (1 + q * q / n)
    return centre - half, centre + half
print()
for k, n in ((70, 100), (700, 1000), (7000, 10000)):
    lo, hi = wilson(k, n)
    print(f"{k:,} right of {n:,}: accuracy {k / n:.0%}, 95% interval {lo:.1%} to {hi:.1%} "
          f"(about ±{(hi - lo) / 2 * 100:.1f} points)")

# What matters for a comparison is the interval on the difference.
n = 500
unpaired = za * math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
paired = za * math.sqrt((0.15 - (p2 - p1) ** 2) / n)
print(f"\n{n} questions, 95% interval on the difference in accuracy: unpaired ±{unpaired * 100:.1f} points, "
      f"paired (disagreeing on 15%) ±{paired * 100:.1f} points")

# Cohen's kappa: agreement between two annotators beyond what chance gives.
# 100 answers labeled good/bad by two people:
#                  B good   B bad
#   A good           45       10
#   A bad             5       40
a = [[45, 10], [5, 40]]
n = sum(map(sum, a))
p_obs = (a[0][0] + a[1][1]) / n
pa_good, pb_good = sum(a[0]) / n, (a[0][0] + a[1][0]) / n
p_chance = pa_good * pb_good + (1 - pa_good) * (1 - pb_good)
kappa = (p_obs - p_chance) / (1 - p_chance)
print(f"\nCohen's kappa: observed agreement {p_obs:.2f}, chance agreement {pa_good:.2f} x {pb_good:.2f} + "
      f"{1 - pa_good:.2f} x {1 - pb_good:.2f} = {p_chance:.3f}, kappa = ({p_obs:.2f} - {p_chance:.3f}) / (1 - {p_chance:.3f}) = {kappa:.2f}")