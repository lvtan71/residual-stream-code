import numpy as np

# CLIP's loss on a batch of 3 image-caption pairs. Embeddings are unit
# vectors; S[i, j] is the cosine similarity of image i and caption j, and the
# logits are S divided by a temperature. The matching pairs are on the diagonal.
S = np.array([[0.90, 0.20, 0.10],
              [0.30, 0.80, 0.40],
              [0.05, 0.35, 0.70]])
t = 0.07                                   # CLIP's initial temperature
L = S / t
def log_softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))
img_to_txt = -np.diag(log_softmax(L, axis=1))     # each image: which caption is mine? (cross-entropy over the row)
txt_to_img = -np.diag(log_softmax(L, axis=0))     # each caption: which image is mine? (over the column)
print("logits = similarities / 0.07:\n", L.round(2))
print("image -> caption losses:", img_to_txt.round(4), " caption -> image losses:", txt_to_img.round(4))
print(f"CLIP loss (symmetric InfoNCE) = mean of both directions = {(img_to_txt.mean() + txt_to_img.mean()) / 2:.4f}")

# SigLIP: no softmax over the batch. Every pair is its own binary question,
# match (label +1) or not (-1), scored with a sigmoid; a learned scale t
# (from log t, initialized at log 10) and bias b (initialized at -10).
tp, b = 10.0, -10.0
z = np.where(np.eye(3) == 1, 1, -1)
pair = np.log1p(np.exp(-z * (tp * S + b)))        # -log sigmoid(z * logit)
print("\nSigLIP per-pair losses:\n", pair.round(4))
print(f"SigLIP loss = sum over all 9 pairs / 3 images = {pair.sum() / 3:.4f}")
