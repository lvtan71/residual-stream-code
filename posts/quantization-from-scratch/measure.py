import numpy as np

# Quantizing real layers: Qwen2.5-0.5B's MLP down-projections (896 x 4,864)
# in layers 12 and 22, with their real inputs from this blog's text. Quality is the
# layer's output error on 2,048 held-out tokens, relative to the output's
# size: ||X W^T - X Q^T|| / ||X W^T||. Calibration (GPTQ, AWQ) uses
# 32,768 other tokens, kept as H = 2 X^T X and each channel's mean |x|
# (capture.py), or, to show what too little does, 2,048.
def quantize(w, bits, axis=None):
    """Asymmetric round-to-nearest: an integer grid 0 .. 2^bits - 1, a scale
    and a zero-point so the grid spans [min, max] along `axis`."""
    lo, hi = w.min(axis=axis, keepdims=True), w.max(axis=axis, keepdims=True)
    scale = (hi - lo) / (2 ** bits - 1)
    scale = np.where(scale == 0, 1, scale)
    zero = np.round(-lo / scale)
    q = np.clip(np.round(w / scale) + zero, 0, 2 ** bits - 1)
    return (q - zero) * scale

def symmetric(w, bits, axis):
    """Symmetric (absmax): a grid centred on zero, -(2^(b-1) - 1) .. 2^(b-1) - 1,
    scale = max|w| / that; no zero-point to store."""
    top = 2 ** (bits - 1) - 1
    scale = np.abs(w).max(axis=axis, keepdims=True) / top
    scale = np.where(scale == 0, 1, scale)
    return np.clip(np.round(w / scale), -top, top) * scale

# NF4 (Dettmers et al., 2023): 16 levels at quantiles of a normal
# distribution, scaled per block by the block's absmax (QLoRA's values).
NF4 = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                0.7229568362236023, 1.0])
def nf4(w, axis):
    m = np.abs(w).max(axis=axis, keepdims=True)
    m = np.where(m == 0, 1, m)
    x = w / m
    return NF4[np.abs(x[..., None] - NF4).argmin(-1)] * m

def rtn(W, bits, granularity):
    if granularity == "per tensor":
        return quantize(W, bits)
    if granularity == "per channel":
        return quantize(W, bits, axis=1)                      # one scale per output row
    G = W.reshape(W.shape[0], -1, 128)                        # groups of 128 inputs per row
    return quantize(G, bits, axis=2).reshape(W.shape)

def gptq(W, bits, H, block=128, damp=0.01):
    """GPTQ (Frantar et al., 2023): quantize one input column at a time, in
    groups of 128, and push each column's rounding error onto the columns
    not yet quantized, weighted by the inverse Hessian of the layer's
    squared output error, H = 2 X^T X."""
    W = W.copy(); n = W.shape[1]
    H = H.copy()
    H += damp * np.mean(np.diag(H)) * np.eye(n)
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T              # upper Cholesky factor of H^-1
    Q = np.zeros_like(W)
    for i1 in range(0, n, block):
        i2 = i1 + block
        W1, E1 = W[:, i1:i2].copy(), np.zeros((W.shape[0], block))
        lo, hi = W1.min(1), W1.max(1)                             # this group's grid, from current weights
        scale = np.where(hi > lo, (hi - lo) / (2 ** bits - 1), 1); zero = np.round(-lo / scale)
        for i in range(block):
            col = W1[:, i]
            q = (np.clip(np.round(col / scale) + zero, 0, 2 ** bits - 1) - zero) * scale
            Q[:, i1 + i] = q
            e = (col - q) / Hinv[i1 + i, i1 + i]
            W1[:, i:] -= np.outer(e, Hinv[i1 + i, i1 + i:i2])     # compensate within the block
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]                        # and in the columns after it
    return Q

def awq(W, bits, H, mag, grid=np.linspace(0, 1, 11)):
    """AWQ (Lin et al., 2024): scale input channel j of the weights up by
    s_j = mean|x_j|^alpha and its inputs down by the same factor, so the
    channels that matter most get finer steps; alpha chosen on calibration
    data. The scaling is exact in full precision."""
    best = None
    for a in grid:
        s = mag ** a; s /= np.sqrt(s.max() * s.min())
        Q = rtn(W * s, bits, "group 128") / s
        D = W - Q
        e = np.sum((D @ H) * D)                                   # calibration output error, from H
        if best is None or e < best[0]:
            best = (e, a, Q)
    return best[2], best[1]

def awq_clip(W, bits, H, mag, alpha, n_grid=20, max_shrink=0.5):
    """The second half of AWQ: after the scaling, for every group of 128
    weights in a row, search how far to clip its range before rounding
    (shrinking it by up to half), keeping whichever gives the least output
    error, here measured with the group's block of H."""
    s = mag ** alpha; s /= np.sqrt(s.max() * s.min())
    Ws = W * s
    Hs = H / s[:, None] / s[None, :]                               # H for the scaled inputs
    Q = np.zeros_like(Ws)
    for g in range(0, Ws.shape[1], 128):
        w, h = Ws[:, g:g + 128], Hs[g:g + 128, g:g + 128]
        top = np.abs(w).max(1, keepdims=True)
        best_err, best_q = np.full(w.shape[0], np.inf), np.zeros_like(w)
        for i in range(n_grid):
            t = top * (1 - max_shrink * i / n_grid)
            q = quantize(np.clip(w, -t, t), bits, axis=1)
            d = q - w
            e = np.sum((d @ h) * d, axis=1)
            better = e < best_err
            best_err[better], best_q[better] = e[better], q[better]
        Q[:, g:g + 128] = best_q
    return Q / s

# Worked: 8 weights to 4 bits.
w = np.array([-0.31, -0.12, 0.0, 0.04, 0.09, 0.18, 0.22, 0.47])
s, z = (w.max() - w.min()) / 15, np.round(-w.min() / ((w.max() - w.min()) / 15))
print(f"worked, 8 weights to 4 bits: scale {s:.4f}, zero-point {z:.0f}")
print(f"  integers {np.clip(np.round(w / s) + z, 0, 15).astype(int).tolist()}")
print(f"  back     {np.round(quantize(w, 4), 3).tolist()}")


for L in (12, 22):
    X, W = np.load(f"X_{L}.npy").astype(np.float64), np.load(f"W_{L}.npy").astype(np.float64)
    Xcal, Xtest = X[:2048], X[2048:]
    H_big, mag_big = np.load(f"H_big_{L}.npy"), np.load(f"mag_big_{L}.npy")
    ref = Xtest @ W.T
    err = lambda Q: np.linalg.norm(ref - Xtest @ Q.T) / np.linalg.norm(ref)
    print(f"\n=== layer {L}: input channels' mean |x|: median {np.median(mag_big):.3f}, largest "
          f"{mag_big.max():.2f} ({mag_big.max() / np.median(mag_big):.0f}x); top 1% of channels carry "
          f"{np.sort(mag_big)[-49:].sum() / mag_big.sum():.0%} of the total")
    print("round-to-nearest, output error (held-out tokens):")
    print(f"{'':12s}" + "".join(f"{b:>8d} bits" for b in (8, 4, 3)))
    for g in ("per tensor", "per channel", "group 128"):
        print(f"{g:12s}" + "".join(f"{err(rtn(W, b, g)):13.2%}" for b in (8, 4, 3)))
    G64 = W.reshape(W.shape[0], -1, 64)
    print(f"4 bits, blocks of 64: asymmetric {err(quantize(G64, 4, axis=2).reshape(W.shape)):.2%}, "
          f"symmetric {err(symmetric(G64, 4, 2).reshape(W.shape)):.2%}, NF4 {err(nf4(G64, 2).reshape(W.shape)):.2%}")
    print("group 128, output error (held-out tokens):")
    H_small = 2 * Xcal.T @ Xcal
    for b in (4, 3):
        Qa, a = awq(W, b, H_big, mag_big)
        print(f"  {b} bits: round-to-nearest {err(rtn(W, b, 'group 128')):.2%}; GPTQ {err(gptq(W, b, H_big)):.2%}; "
              f"AWQ scaling only {err(Qa):.2%} (alpha {a:.1f}), with clipping {err(awq_clip(W, b, H_big, mag_big, a)):.2%}")
    print("  GPTQ at 4 bits with 2,048 calibration tokens, by damping (fraction of H's mean diagonal): " +
          ", ".join(f"{d:g}: {err(gptq(W, 4, H_small, damp=d)):.2%}" for d in (0.01, 0.1, 1.0)) +
          f"; on its own calibration tokens (0.01): "
          f"{np.linalg.norm(Xcal @ W.T - Xcal @ gptq(W, 4, H_small).T) / np.linalg.norm(Xcal @ W.T):.2%}")