# Compute-optimal training from a fitted scaling law,
#     L(N, D) = E + A / N^alpha + B / D^beta,
# the loss of a model with N parameters trained on D tokens (here D is
# tokens, not width), with training compute C = 6 N D ("Backpropagation, by Hand"). Three
# versions of one fit: Hoffmann et al.'s (2022, "Chinchilla", approach 3) as
# printed in the paper, the same with the unrounded constants from the
# paper's source, and Besiroglu et al.'s (2024) refit of the same data.
FITS = {"Hoffmann, as printed": dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28),
        "Hoffmann, unrounded": dict(E=1.6934, A=406.4, B=410.7, alpha=0.3392, beta=0.2849),
        "Besiroglu refit": dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)}

def loss(f, N, D):
    return f["E"] + f["A"] / N ** f["alpha"] + f["B"] / D ** f["beta"]

def optimal(f, C):
    """Minimizing L subject to 6ND = C. Setting dL/dN = 0 along the
    constraint gives alpha A / N^alpha = beta B / D^beta, so
    N* = G (C/6)^(beta/(alpha+beta)), G = (alpha A / (beta B))^(1/(alpha+beta))."""
    a, b = f["alpha"], f["beta"]
    G = (a * f["A"] / (b * f["B"])) ** (1 / (a + b))
    N = G * (C / 6) ** (b / (a + b))
    return N, C / 6 / N

for name, f in FITS.items():
    rows = []
    for C in (1e21, 1e23, 1e25):
        N, D = optimal(f, C)
        rows.append(f"{D / N:5.1f}")
    print(f"{name:21s} tokens per parameter at C = 1e21 / 1e23 / 1e25 FLOPs: {' / '.join(rows)}")

# Llama 3 8B: about 8 billion parameters, trained on about 15 trillion tokens.
N8, D8 = 8e9, 15e12
C = 6 * N8 * D8
print(f"\nLlama 3 8B: {D8 / N8:,.0f} tokens per parameter, C = {C:.1e} FLOPs")
for name, f in FITS.items():
    No, Do = optimal(f, C)
    target = loss(f, N8, D8)
    lo, hi = 1e18, C                                    # least compute whose optimal model reaches the 8B's loss
    for _ in range(100):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if loss(f, *optimal(f, mid)) > target else (lo, mid)
    Nm, _ = optimal(f, hi)
    print(f"  {name:21s} same compute: optimal {No / 1e9:3.0f}B on {Do / 1e12:4.1f}T tokens, loss {loss(f, No, Do):.3f} "
          f"vs the 8B's {target:.3f}; same loss: a {Nm / 1e9:.0f}B model with {hi / C:.2f}x the 8B's training compute")

# For the chart: loss against model size at the 8B's training compute.
for name, f in FITS.items():
    pts = [(n, loss(f, n * 1e9, C / 6 / (n * 1e9))) for n in (2, 4, 8, 16, 32, 64, 128, 256)]
    print(f"curve {name}: " + " ".join(f"{n}:{l:.4f}" for n, l in pts))
