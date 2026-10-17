# optim.py
import numpy as np

class SGD:
    def __init__(self, eta):
        self.eta = eta
    def step(self, p, g, key):
        p -= self.eta * g

class Momentum:
    def __init__(self, eta, beta=0.9):
        self.eta, self.beta, self.v = eta, beta, {}
    def step(self, p, g, key):
        v = self.v[key] = self.beta * self.v.get(key, 0) + g
        p -= self.eta * v

def decays(key):
    """Weight decay applies to weight matrices only, not to biases or norm gains."""
    name = key[-1] if isinstance(key, tuple) else key
    return name == "W"

class Adam:
    def __init__(self, eta, beta1=0.9, beta2=0.999, eps=1e-8, decay=0.0, decoupled=True):
        self.eta, self.b1, self.b2, self.eps = eta, beta1, beta2, eps
        self.decay, self.decoupled, self.m, self.v, self.t = decay, decoupled, {}, {}, {}
    def step(self, p, g, key):
        if self.decay and not self.decoupled and decays(key):
            g = g + self.decay * p                              # L2 folded into the gradient
        t = self.t[key] = self.t.get(key, 0) + 1
        m = self.m[key] = self.b1 * self.m.get(key, 0) + (1 - self.b1) * g
        v = self.v[key] = self.b2 * self.v.get(key, 0) + (1 - self.b2) * g * g
        m_hat, v_hat = m / (1 - self.b1 ** t), v / (1 - self.b2 ** t)   # bias correction
        if self.decay and self.decoupled and decays(key):
            p -= self.eta * self.decay * p                      # AdamW: decay applied directly
        p -= self.eta * m_hat / (np.sqrt(v_hat) + self.eps)
