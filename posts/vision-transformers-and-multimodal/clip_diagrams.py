import glob
import re
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# CLIP (ViT-B/32) on this blog's own diagrams. Each diagram's caption is the
# alt text its post gives it. Can CLIP find the right diagram from its
# caption, and the right caption from the diagram, among the 19? The
# diagrams are rendered whole into 512x512 squares, once as they are and
# once with every text label hidden: render.py on a Mac, or unpack
# clip-diagram-renders.tar.gz. The alt texts come from the blog snapshot
# (tar xzf blog-11f17f9.tar.gz).
alts = {}
for path in glob.glob("content/posts/*.mdx"):
    for m in re.finditer(r'light="/images/diagrams/([^"]+)-light\.svg"\s+dark="[^"]+"\s+alt="([^"]+)"', open(path).read()):
        alts[m.group(1)] = m.group(2)
names = sorted(n for n in alts if glob.glob(f"fit/{n}.png"))
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268").eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
with torch.no_grad():
    def image(folder):               # each tower's pooled output, then CLIP's projection into the shared space
        px = proc(images=[Image.open(f"{folder}/{n}.png").convert("RGB") for n in names], return_tensors="pt")
        return model.visual_projection(model.vision_model(pixel_values=px["pixel_values"]).pooler_output)
    def text(captions):              # CLIP reads at most 77 tokens; longer alt texts are cut
        enc = proc(text=captions, return_tensors="pt", padding=True, truncation=True)
        out = model.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return model.text_projection(out.pooler_output)
    caps = {"the post's alt text": [alts[n] for n in names],
            "the file name as words": [n.replace("-", " ") for n in names]}
    print(f"{len(names)} diagrams; chance of the right match at random: 1 in {len(names)} ({1 / len(names):.0%})\n")
    print(f"{'diagrams':20s} {'caption':24s} {'caption -> diagram':>19s} {'diagram -> caption':>19s}  (right one first)")
    for folder, label in (("fit", "as drawn"), ("notext", "with labels hidden")):
        img = image(folder)
        for name, c in caps.items():
            S = torch.nn.functional.normalize(img, dim=-1) @ torch.nn.functional.normalize(text(c), dim=-1).T
            i2t = int((S.argmax(1) == torch.arange(len(names))).sum())
            t2i = int((S.argmax(0) == torch.arange(len(names))).sum())
            print(f"{label:20s} {name:24s} {t2i:>12d} of {len(names)} {i2t:>12d} of {len(names)}")
    n_tok = [len(proc.tokenizer(a).input_ids) for a in caps["the post's alt text"]]
    print(f"\nalt texts: median {sorted(n_tok)[len(n_tok) // 2]} tokens, {sum(x > 77 for x in n_tok)} of {len(n_tok)} cut at 77")
