# Seeing with Transformers: the full code

The scripts behind [Seeing with Transformers: ViT, CLIP and Vision-Language Models](https://residual-stream.lvtan71.workers.dev/posts/vision-transformers-and-multimodal/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `patches.py` | A 224x224 image as ViT-B/16 patch tokens: patch count, patch-embedding parameters, sequence length at several resolutions, and the KV cache one image costs in LLaVA-1.5. | Python, instant |
| `clip_loss.py` | CLIP's symmetric InfoNCE loss and SigLIP's pairwise sigmoid loss on a batch of three image-caption pairs. | CPU, instant |
| `render.py` | Renders each of the blog's diagrams into a 512x512 square with headless Chrome, as drawn (`fit/`) and with all text hidden (`notext/`). **macOS only** (Chrome's macOS install path). Its output is shipped as `clip-diagram-renders.tar.gz`, so other machines skip it. Not in the run list below. | Mac with Google Chrome, about a minute |
| `clip_diagrams.py` | CLIP ViT-B/32 matching 19 diagrams to their posts' alt texts and to their file names, as drawn and with labels hidden, in both directions. | CPU, about 2 minutes; downloads `openai/clip-vit-base-patch32` (about 600 MB) at a pinned revision |

`blog-11f17f9.tar.gz` links to `../../data/blog-11f17f9.tar.gz`: the blog's posts and diagrams at commit 11f17f9, where `clip_diagrams.py` reads the alt texts and `render.py` the SVGs. `clip-diagram-renders.tar.gz` links to `../../data/clip-diagram-renders.tar.gz`: the `fit/` and `notext/` PNGs that `render.py` made for the post.

The whole folder runs in about 2 minutes on a CPU. The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17.

```bash
pip install numpy "torch==2.14.*" "transformers==5.17.*" pillow huggingface_hub
tar xzf blog-11f17f9.tar.gz
tar xzf clip-diagram-renders.tar.gz
python patches.py
python clip_loss.py
python clip_diagrams.py
```
