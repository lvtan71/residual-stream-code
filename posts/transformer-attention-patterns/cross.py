import json
import torch
from huggingface_hub import snapshot_download
from transformers import MarianMTModel, MarianTokenizer

# Cross-attention in a real encoder-decoder: Marian's English-to-German
# translation model (6 encoder and 6 decoder layers, 8 heads). Translate one
# sentence, then run the decoder over its own output and read its
# cross-attention: for each German token, the weights it puts on the English
# tokens, averaged over the heads of one layer. Printed: the English token
# each German token weights most, per layer; the weights of the layer whose
# picks fall closest to the diagonal of a word-for-word alignment are saved
# to cross.json for the post's figure. CPU is enough.
PATH = snapshot_download("Helsinki-NLP/opus-mt-en-de", revision="6183067f769a302e3861815543b9f312c71b0ca4",
                         allow_patterns=["*.json", "*.spm", "pytorch_model.bin"])
tok = MarianTokenizer.from_pretrained(PATH)
model = MarianMTModel.from_pretrained(PATH, attn_implementation="eager").eval()
text = "The black cat sleeps on the warm mat in the kitchen."
src = tok(text, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**src, num_beams=1, do_sample=False, max_new_tokens=40)
    a = model(**src, decoder_input_ids=out[:, :-1], output_attentions=True)
en = tok.convert_ids_to_tokens(src.input_ids[0])
de = tok.convert_ids_to_tokens(out[0, 1:])                  # the token each decoder step produced
print("English:", text)
print("German: ", tok.decode(out[0], skip_special_tokens=True))
print(f"{len(de)} German tokens attend over {len(en)} English tokens (including </s>)")
best, best_layer = None, None
for layer, w in enumerate(a.cross_attentions):
    w = w[0].mean(0)                                         # [German, English], averaged over 8 heads
    picks = w.argmax(-1).tolist()
    spread = sum(abs(p / (len(en) - 1) - i / (len(de) - 1)) for i, p in enumerate(picks)) / len(picks)
    print(f"layer {layer}: each German token's most-weighted English token: " +
          " ".join(f"{d}->{en[p]}" for d, p in zip(de, picks)))
    if best is None or spread < best:
        best, best_layer, weights = spread, layer, w
print(f"layer {best_layer} is closest to a word-for-word diagonal")
json.dump({"layer": best_layer, "en": en, "de": de, "weights": weights.tolist()}, open("cross.json", "w"))
