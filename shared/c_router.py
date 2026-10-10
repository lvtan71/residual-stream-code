import re
import numpy as np

SERVING = {"kv", "cache", "caching", "batch", "batching", "latency",
           "throughput", "serve", "serving", "vllm", "sglang", "tensorrt",
           "gpu", "decode", "prefill", "paged", "quantization", "memory"}

questions = [  # (question, 1 = send to the serving posts)
    ("How big is the KV cache for a 70B model?", 1),
    ("Why is decode slower per token than prefill?", 1),
    ("Does vLLM use paged attention?", 1),
    ("How does continuous batching raise throughput?", 1),
    ("What limits the batch size on one GPU?", 1),
    ("Is SGLang faster than TensorRT-LLM?", 1),
    ("Why is my chatbot slow to answer long prompts?", 1),
    ("How much memory does speculative decoding save?", 1),
    ("What does quantization do to latency?", 1),
    ("Why does the first token take so long?", 1),
    ("How do I serve a model to many users?", 1),
    ("Where does prefix caching help?", 1),
    ("How does backpropagation work?", 0),
    ("What is a residual stream?", 0),
    ("Why scale attention by the square root of d_k?", 0),
    ("How does BPE pick which pair to merge?", 0),
    ("What is the difference between an agent and a workflow?", 0),
    ("How does the memory tool in context engineering work?", 0),
    ("How much GPU memory does training a 7B model need?", 0),
    ("What does compaction keep from the transcript?", 0),
    ("Why did the chess model need a leakage contract?", 0),
    ("How is softmax computed?", 0),
    ("What does the tokenizer do with emoji?", 0),
    ("Should I cache the embeddings of every post?", 0),
]

def hits(q):
    return sum(tok in SERVING for tok in re.findall(r"[a-z0-9]+", q.lower()))

x = np.array([hits(q) for q, _ in questions], dtype=float)
y = np.array([label for _, label in questions], dtype=float)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    w, b, eta = 0.0, 0.0, 0.5
    for step in range(5000):
        p = sigmoid(w * x + b)
        w -= eta * ((p - y) * x).mean()      # the whole gradient: (p - y) x
        b -= eta * (p - y).mean()

    p = sigmoid(w * x + b)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    print(f"w = {w:.3f}, b = {b:.3f}, log-loss = {loss:.4f}")
    for k in range(3):
        print(f"P(serving | {k} keyword hits) = {sigmoid(w * k + b):.3f}")
    print(f"accuracy = {((p >= 0.5) == y).mean():.3f}")
