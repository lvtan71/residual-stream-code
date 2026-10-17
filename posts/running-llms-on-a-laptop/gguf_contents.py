import collections
import os
from gguf import GGUFReader
from huggingface_hub import snapshot_download

# What is inside a GGUF file: a header, key-value metadata (architecture,
# hyperparameters, the tokenizer itself), then a table of tensors, each
# with a name, shape, type and offset, then the tensor data, aligned so it
# can be memory-mapped straight from disk.
D = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                      revision="91cad51170dc346986eccefdc2dd33a9da36ead9",
                      allow_patterns=["*q3_k_m.gguf", "*q4_k_m.gguf", "*q8_0.gguf"]) + "/"
r = GGUFReader(D + "qwen2.5-1.5b-instruct-q4_k_m.gguf")
fields = r.fields
def value(key):
    f = fields[key]
    return f.parts[f.data[0]].tobytes().decode() if f.types and f.types[0].name == "STRING" else f.parts[f.data[0]][0]
kv = [k for k in fields if not k.startswith("GGUF.")]      # the reader adds GGUF.version and the counts itself
print(f"qwen2.5-1.5b-instruct-q4_k_m.gguf: {len(kv)} metadata keys, {len(r.tensors)} tensors")
for key in ("general.architecture", "qwen2.block_count", "qwen2.context_length", "qwen2.embedding_length",
            "qwen2.attention.head_count", "qwen2.attention.head_count_kv", "general.file_type"):
    print(f"  {key:32s} {value(key)}")
print(f"  {'tokenizer.ggml.tokens':32s} {len(fields['tokenizer.ggml.tokens'].data):,} entries (the vocabulary, in the file)")
by_type = collections.defaultdict(lambda: [0, 0, 0])
for t in r.tensors:
    b = by_type[t.tensor_type.name]
    b[0] += 1; b[1] += int(t.n_elements); b[2] += int(t.n_bytes)
print("tensors by type:")
for name, (n, elems, nbytes) in sorted(by_type.items(), key=lambda kv: -kv[1][2]):
    print(f"  {name:6s} {n:4d} tensors {elems / 1e6:8.1f}M weights {nbytes / 2**20:8.1f} MiB  {8 * nbytes / elems:5.2f} bits per weight")
print("examples:", ", ".join(f"{t.name} {t.tensor_type.name}" for t in r.tensors[:6]))
q6 = [t.name for t in r.tensors if t.tensor_type.name == "Q6_K"]
print(f"the {len(q6)} Q6_K tensors: output.weight, {sum('attn_v' in n for n in q6)} attn_v and "
      f"{sum('ffn_down' in n for n in q6)} ffn_down (layers {sorted({int(n.split('.')[1]) for n in q6 if n.startswith('blk')})})")
for f in sorted(os.listdir(D)):
    if f.endswith(".gguf"):
        rr = GGUFReader(D + f)
        n = sum(int(t.n_elements) for t in rr.tensors)
        print(f"{f:36s} {os.path.getsize(D + f) / 1e9:5.2f} GB, {8 * sum(int(t.n_bytes) for t in rr.tensors) / n:5.2f} bits per weight on average")
