import numpy as np

class Tensor:
    """A NumPy array that remembers how it was made, so it can run backward."""
    def __init__(self, data, parents=(), backward=None):
        self.data, self.grad = np.asarray(data, dtype=float), None
        self.parents, self._backward = parents, backward   # the tape: one node per operation

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other))
        def backward(g):                                    # sum over broadcast dimensions
            return unbroadcast(g, self.data.shape), unbroadcast(g, other.data.shape)
        out._backward = backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))
        out._backward = lambda g: (g @ other.data.T, self.data.T @ g)   # "Backpropagation, by Hand"'s rules
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,))
        out._backward = lambda g: (g * (self.data > 0),)
        return out

    def bce_with_logits(self, y):
        """Mean log-loss of sigmoid(self) against labels y, as one operation.
        (Not stable for very large logits; PyTorch's uses log-sum-exp, as in "Cross-Entropy and the Softmax Gradient".)"""
        p = 1 / (1 + np.exp(-self.data))
        out = Tensor(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(), (self,))
        out._backward = lambda g: (g * (p - y) / y.size,)             # p - y, "Learning Is Optimization"
        return out

    def backward(self):
        order, seen = [], set()
        def visit(t):                                        # topological order of the tape
            if id(t) not in seen:
                seen.add(id(t))
                for p in t.parents:
                    visit(p)
                order.append(t)
        visit(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):                            # from the loss back to the inputs
            if t._backward:
                for p, g in zip(t.parents, t._backward(t.grad)):
                    p.grad = g if p.grad is None else p.grad + g   # a tensor used twice sums its gradients

def unbroadcast(g, shape):
    while g.ndim > len(shape):                               # dimensions added in front
        g = g.sum(0)
    for i, s in enumerate(shape):                            # dimensions stretched from size 1
        if s == 1:
            g = g.sum(i, keepdims=True)
    return g
