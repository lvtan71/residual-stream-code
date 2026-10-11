import numpy as np

# Softmax on the post's toy logits, and its two properties: adding a constant
# to every logit changes nothing, and multiplying them sharpens or flattens.
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

def softmax(z):
    e = np.exp(z - z.max())                 # subtracting the max changes nothing, and avoids overflow
    return e / e.sum()

z = np.array([0.4, 1.1, -0.4, -1.1])        # logits for "the", "cat", "sat", "."
print("exp(z):          ", np.exp(z), " sum", f"{np.exp(z).sum():.3f}")
print("softmax(z):      ", softmax(z))
print("softmax(z + 10): ", softmax(z + 10))
for s in (0.5, 2, 10):
    print(f"softmax({s:g} * z): ".ljust(18), softmax(s * z))
