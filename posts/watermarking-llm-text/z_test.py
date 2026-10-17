import math
# Detection is a one-proportion z-test. Without a watermark each scored
# token is green with probability gamma, independently (the list depends on
# a secret key), so in T tokens the green count has mean gamma T and
# variance T gamma (1 - gamma):
#   z = (greens - gamma T) / sqrt(T gamma (1 - gamma)).
gamma, delta, T = 0.25, 2.0, 200
# If the model were indifferent between all tokens, adding delta to the
# green logits would make a token green with probability
#   gamma e^delta / (gamma e^delta + 1 - gamma),
# the most the watermark can do. Low-entropy text gives less.
g_max = gamma * math.exp(delta) / (gamma * math.exp(delta) + 1 - gamma)
z = lambda g, T: (g - gamma) * T / math.sqrt(T * gamma * (1 - gamma))
print(f"gamma = {gamma}, delta = {delta}: a flat distribution's green share is {g_max:.3f}")
print(f"z for T = {T} tokens at that share: {z(g_max, T):.1f}")
print(f"\n{'green share':>11s} {'z at T=50':>10s} {'z at T=200':>11s} {'tokens for z = 4':>17s}")
for g in (0.30, 0.35, 0.40, 0.50, g_max):
    need = 16 * gamma * (1 - gamma) / (g - gamma) ** 2
    print(f"{g:11.3f} {z(g, 50):10.1f} {z(g, 200):11.1f} {math.ceil(need):17d}")
# A false positive needs z > 4 on unwatermarked text. The normal tail is the
# large-T limit; the green count is really binomial, with a fatter tail.
print(f"\nfalse-positive rate at z > 4: normal tail {0.5 * math.erfc(4 / math.sqrt(2)):.1e}", end="")
for T in (50, 200, 1000):
    k = math.floor(gamma * T + 4 * math.sqrt(T * gamma * (1 - gamma))) + 1      # fewest greens with z > 4
    exact = sum(math.comb(T, i) * gamma ** i * (1 - gamma) ** (T - i) for i in range(k, T + 1))
    print(f"; exact at T = {T}: {exact:.1e}", end="")
print()