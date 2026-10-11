import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# The model for this post: Qwen2.5-1.5B-Instruct, in 16-bit floats on the
# laptop GPU.
PATH = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                         revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float16).to(DEVICE).eval()
encode = lambda text: torch.tensor(tok(text).input_ids, device=DEVICE)
