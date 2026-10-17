import time
from gguf import GGUFReader
from llama_cpp import Llama
from huggingface_hub import snapshot_download

# Tokens per second, measured against the bandwidth bound of "Small and Specialized Models":
# each decoded token reads every weight once, so tokens/s <= bandwidth /
# bytes of weights. The bytes read per token are every tensor except the
# token-embedding table, of which a token reads one row. This laptop's GPU
# measured 93 GB/s in "How a GPU Runs Deep Learning". Decode
# speed is timed from the first generated token to the last (128 tokens,
# greedy), so the prompt's prefill is excluded; median of 3 runs. The model
# is locked in RAM (use_mlock), so the OS can't page it out mid-run.
D = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                      revision="91cad51170dc346986eccefdc2dd33a9da36ead9",
                      allow_patterns=["*q3_k_m.gguf", "*q4_k_m.gguf", "*q8_0.gguf"]) + "/"
BW = 93e9
prompt = "<|im_start|>user\nExplain what a KV cache is, in detail.<|im_end|>\n<|im_start|>assistant\n"
def bytes_per_token(path):
    tensors = GGUFReader(path).tensors
    emb = next(t for t in tensors if t.name == "token_embd.weight")
    return sum(int(t.n_bytes) for t in tensors) - int(emb.n_bytes) * (1 - 1 / emb.shape[-1])

print(f"{'file':36s} {'GB read/token':>13s} {'bound':>9s} {'GPU (Metal)':>12s} {'of bound':>9s} {'CPU, 8 threads':>15s}")
for f in ("qwen2.5-1.5b-instruct-q8_0.gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf", "qwen2.5-1.5b-instruct-q3_k_m.gguf"):
    size = bytes_per_token(D + f)
    speeds = {}
    for name, layers in (("gpu", -1), ("cpu", 0)):
        llm = Llama(D + f, n_gpu_layers=layers, n_ctx=1024, n_threads=8, verbose=False, seed=0, use_mlock=True)
        runs = []
        for _ in range(3):
            stamps = []
            for _ in llm(prompt, max_tokens=128, temperature=0.0, top_k=1, repeat_penalty=1.0, stream=True):
                stamps.append(time.perf_counter())
            runs.append((len(stamps) - 1) / (stamps[-1] - stamps[0]))
        speeds[name] = sorted(runs)[1]
        del llm
    print(f"{f:36s} {size / 1e9:11.2f}   {BW / size:6.0f}/s {speeds['gpu']:9.0f}/s {speeds['gpu'] / (BW / size):8.0%} "
          f"{speeds['cpu']:12.0f}/s")
