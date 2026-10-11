# The KV cache's memory formula, and an idealized model of decoding on one
# H100 SXM (3.35 TB/s, 80 GB), following Stanford CS336's inference lecture:
# each decode step reads every weight once and every sequence's whole cache
# once, and nothing else limits it.
#   cache bytes per token = 2 (K and V) x layers x kv heads x head dim x bytes
def per_token(layers, kv_heads, head_dim, bytes_=2):
    return 2 * layers * kv_heads * head_dim * bytes_

print("KV cache per token, 16-bit:")
for name, b in (("Llama 2 7B, 32 KV heads", per_token(32, 32, 128)),
                ("Llama 2 70B, GQA, 8 KV heads", per_token(80, 8, 128)),
                ("Llama 2 70B if it had 64 KV heads", per_token(80, 64, 128)),
                ("Qwen2.5-1.5B, GQA, 2 KV heads", per_token(28, 2, 128)),
                ("DeepSeek-V2, MLA, 512 + 64 cached", 60 * (512 + 64) * 2)):
    print(f"  {name:36s} {b / 1024:7.1f} KiB")

def params(L, D, F, vocab, kv_dim):
    """Llama-style: untied embedding and unembedding; per layer Q and O are
    D x D, K and V are D x kv_dim, a SwiGLU MLP of width F, two norms."""
    return 2 * vocab * D + L * (2 * D * D + 2 * D * kv_dim + 3 * D * F + 2 * D) + D

BW, HBM, CONTEXT = 3.35e12, 80e9, 1024
print("\nLlama 2 13B on one H100, 1,024-token contexts, bandwidth-bound decode:")
print("KV heads  batch   weights   cache/seq   memory     latency     throughput")
for kv_heads in (40, 8):
    W = params(40, 5120, 13824, 32000, kv_heads * 128) * 2               # 16-bit weights
    cache = per_token(40, kv_heads, 128) * CONTEXT
    for B in (1, 64, 256):
        mem = W + B * cache
        lat = mem / BW
        fits = "" if mem <= HBM else "  does not fit in 80 GB"
        print(f"{kv_heads:8d}  {B:5d}  {W / 1e9:6.1f} GB  {cache / 1e9:7.2f} GB  {mem / 1e9:6.1f} GB  "
              f"{lat * 1e3:6.1f} ms/token  {B / lat:8,.0f} tokens/s{fits}")
