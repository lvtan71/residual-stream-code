"""What shipped: the attention layout of released models, read from their config.json.

Downloads only each model's config.json, pinned to a revision. For models
whose attention layers use a plain grouped-query cache, it also computes the
KV cache that layout needs at 256K tokens in BF16, against the same model
with attention in every layer.
"""

import json

from huggingface_hub import hf_hub_download

REVISIONS = {
    "mistralai/Mistral-7B-v0.1": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
    "mistralai/Mixtral-8x7B-v0.1": "fc7ac94680e38d7348cfa806e51218e6273104b0",
    "NousResearch/Llama-2-7b-hf": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
    "openai/gpt-oss-20b": "6cee5e81ee83917806bbde320786a8fb61efebee",
    "ai21labs/Jamba-v0.1": "9efd11575ba791d9e3d25d4c8b670e78506b2df7",
    "nvidia/Nemotron-H-8B-Base-8K": "94ea861e008c2dfced3e8e1302094024077aa04e",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "9c7f2fbe84465e40164a94cc16cd30b6999b0cc7",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct": "e1df551a447157d4658b573f9a695d57658590e9",
    "MiniMaxAI/MiniMax-Text-01": "a7351bf2bee0e1253919d349f1ad304e6dac13e9",
    "MiniMaxAI/MiniMax-M2": "757303d492a50514c312788b5247a4f696a4c6a3",
    "deepseek-ai/DeepSeek-V3.2-Exp": "194c67e12b1b0d6df0ef373ddcf215bc84027409",
    "deepseek-ai/DeepSeek-V4-Pro": "b5968e9190ef611bbf34a7229255be88a0e937c1",
    "MiniMaxAI/MiniMax-M3": "f0e1c1e04d40177e4673a22097036854f536e9c0",
}
CONTEXT = 256 * 1024


def config(repo):
    with open(hf_hub_download(repo, "config.json", revision=REVISIONS[repo])) as f:
        return json.load(f)


def kv_gb(attn_layers, kv_heads, head_dim, n=CONTEXT):
    """A key and a value per token, per attention layer, per KV head, 2 bytes each."""
    return 2 * attn_layers * kv_heads * head_dim * 2 * n / 2**30


def layout(name, layers, attn, kv_heads=None, head_dim=None, note="", kind="layers"):
    line = f"{name:22s} {attn:>3d} of {layers:>3d} {kind} attention ({attn / layers:4.0%})"
    if kv_heads:
        line += (f", KV cache at 256K {kv_gb(attn, kv_heads, head_dim):6.1f} GiB"
                 f" (all-attention {kv_gb(layers, kv_heads, head_dim):6.1f} GiB)")
    print(line + (f"; {note}" if note else ""))


print("full attention, for comparison")
for repo in ("NousResearch/Llama-2-7b-hf", "mistralai/Mixtral-8x7B-v0.1"):
    c = config(repo)
    hd = c["hidden_size"] // c["num_attention_heads"]
    layout(repo.split("/")[1], c["num_hidden_layers"], c["num_hidden_layers"],
           c["num_key_value_heads"], hd)

print("\nsliding windows")
c = config("mistralai/Mistral-7B-v0.1")
print(f"Mistral-7B-v0.1        sliding_window {c['sliding_window']} in all {c['num_hidden_layers']} layers")
c = config("openai/gpt-oss-20b")
types = c["layer_types"]
print(f"gpt-oss-20b            {types.count('sliding_attention')} sliding ({c['sliding_window']} tokens) and "
      f"{types.count('full_attention')} full layers, alternating: {types[:4]}")

print("\nhybrids: attention beside linear-attention or state-space layers")
c = config("ai21labs/Jamba-v0.1")
L = c["num_hidden_layers"]
attn = sum(1 for i in range(L) if i % c["attn_layer_period"] == c["attn_layer_offset"])
layout("Jamba-v0.1", L, attn, c["num_key_value_heads"], c["hidden_size"] // c["num_attention_heads"],
       f"Mamba state size {c['mamba_d_state']}")

c = config("nvidia/Nemotron-H-8B-Base-8K")
p = c["hybrid_override_pattern"]
mixers = p.count("M") + p.count("*")
layout("Nemotron-H-8B", mixers, p.count("*"), c["num_key_value_heads"], c["attention_head_dim"],
       f"plus {p.count('-')} MLP layers", kind="mixers")

c = config("Qwen/Qwen3-Next-80B-A3B-Instruct")
L = c["num_hidden_layers"]
layout("Qwen3-Next-80B-A3B", L, L // c["full_attention_interval"], c["num_key_value_heads"],
       c["head_dim"], f"one attention layer every {c['full_attention_interval']}, the rest Gated DeltaNet")

c = config("moonshotai/Kimi-Linear-48B-A3B-Instruct")
la = c["linear_attn_config"]
layout("Kimi-Linear-48B-A3B", c["num_hidden_layers"], len(la["full_attn_layers"]),
       note=f"attention (MLA) at layers {la['full_attn_layers']}, Kimi Delta Attention elsewhere")

c = config("MiniMaxAI/MiniMax-Text-01")
t = c["attn_type_list"]
layout("MiniMax-Text-01", len(t), sum(t), c["num_key_value_heads"], c["head_dim"],
       "the rest lightning (linear) attention")

print("\nback to full attention, then sparse")
c = config("MiniMaxAI/MiniMax-M2")
t = c["attn_type_list"]
layout("MiniMax-M2", len(t), sum(t), c["num_key_value_heads"], c["head_dim"])
s = config("MiniMaxAI/MiniMax-M3")["text_config"]["sparse_attention_config"]
f = s["sparse_attention_freq"]
print(f"MiniMax-M3             {sum(f)} of {len(f)} layers sparse: top {s['sparse_topk_blocks']} blocks of "
      f"{s['sparse_block_size']} tokens per query, {s['sparse_topk_blocks'] * s['sparse_block_size']:,} keys; "
      f"the first {f.index(1)} layers full")

print("\nlearned sparse attention")
c = config("deepseek-ai/DeepSeek-V3.2-Exp")
print(f"DeepSeek-V3.2-Exp      indexer picks top {c['index_topk']:,} earlier tokens per query, "
      f"in all {c['num_hidden_layers']} layers")
c = config("deepseek-ai/DeepSeek-V4-Pro")
L, r = c["num_hidden_layers"], c["compress_ratios"]
main = r[:L]                                      # the last entry is the multi-token-prediction layer
print(f"DeepSeek-V4-Pro        {L} layers: {main.count(4)} merge 4 tokens per entry (CSA), "
      f"{main.count(128)} merge 128 (HCA); top {c['index_topk']:,} entries, window {c['sliding_window']}")
