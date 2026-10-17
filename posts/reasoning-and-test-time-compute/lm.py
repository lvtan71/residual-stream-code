import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# The model of "Decoding", Qwen2.5-1.5B-Instruct, loaded on first use, so that code
# needing only the tokenizer doesn't load the weights.
PATH = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                         revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(PATH)
_model = None

def model():
    global _model
    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float16).to(DEVICE).eval()
    return _model
