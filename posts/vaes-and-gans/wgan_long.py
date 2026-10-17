import torch
# WGAN-GP, seed 2, run to 12,000 generator steps and scored every 1,500:
# gen2d.py's gan() with a scoring hook. Scoring draws from the same global
# random generator as the generator's noise, so after the first checkpoint
# this run follows a different path from the one in the table.
src = open("gen2d.py").read().split("# The GAN's value")[0]
src = src.replace("        og.zero_grad(); lg.backward(); og.step()\n",
                  "        og.zero_grad(); lg.backward(); og.step()\n"
                  "        if (step + 1) % 1500 == 0: report(step + 1, G)\n")
ns = {}; exec(src, ns)
def report(step, G):
    with torch.no_grad():
        x = G(torch.randn(2000, 2))
    modes, near = ns["score"](x)
    print(f"step {step:6d}   modes {modes}   near a mode {near:5.1%}   "
          f"sample std ({x[:, 0].std():.2f}, {x[:, 1].std():.2f})", flush=True)
ns["report"] = report
ns["gan"](2, steps=12000, wgan=True)