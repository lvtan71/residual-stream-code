import math
import sys
import numpy as np
from gguf import GGUFReader
from llama_cpp import Llama
from huggingface_hub import snapshot_download

# What each quantization costs in quality: bits per token on two of this
# blog's posts, written after the model was trained (one 1,024-token window
# from the start of each, the second half of each window scored), computed
# from llama.cpp's own logits for each file.
D = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                      revision="91cad51170dc346986eccefdc2dd33a9da36ead9",
                      allow_patterns=["*q3_k_m.gguf", "*q4_k_m.gguf", "*q8_0.gguf"]) + "/"
sys.path.insert(0, ".")
from corpus import clean
POSTS = "content/posts/"
texts = [clean(open(POSTS + f + ".mdx").read()) for f in ("kv-cache-explained", "bpe-tokenizer-explained")]
emb = next(t for t in GGUFReader(D + "qwen2.5-1.5b-instruct-q8_0.gguf").tensors if t.name == "token_embd.weight")
print("token_embd shape in the file:", list(emb.shape))
for f in ("qwen2.5-1.5b-instruct-q8_0.gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf", "qwen2.5-1.5b-instruct-q3_k_m.gguf"):
    llm = Llama(D + f, n_gpu_layers=-1, n_ctx=1024, logits_all=True, verbose=False)
    total, n = 0.0, 0
    for text in texts:
        seq = llm.tokenize(text.encode(), add_bos=False)[:1024]
        llm.reset(); llm.eval(seq)
        logits = np.array(llm.scores[:len(seq)])
        logp = logits - np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1, keepdims=True)) - logits.max(-1, keepdims=True)
        for i in range(512, len(seq)):
            total -= logp[i - 1, seq[i]]; n += 1
    print(f"{f:36s} {total / n / math.log(2):.3f} bits per token")
    del llm
