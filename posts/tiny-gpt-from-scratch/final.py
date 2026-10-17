import sys
import torch
from train import run, final_split, tokenizer_for

# The one scored run: the settings chosen on the dev slice, trained on all
# the training posts, scored on the two held-out posts. Saves the model at
# a few checkpoints for sampling.
seed, schedule = int(sys.argv[1]), sys.argv[2]
vocab = tokenizer_for("char", final_split["fit"])[2]
def save(step, model):
    if seed == 0:
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({"model": state, "config": dict(model.c.__dict__), "vocab": vocab}, f"gpt_{step}.pt")
_, curve, passes = run("char", seed, final_split, 4000, checkpoints=(250, 1000, 2000, 3000),
                       dropout=0.2, schedule=schedule, on_checkpoint=save)
print(f"seed {seed}, {schedule} schedule, held-out bits per byte at "
      + " / ".join(f"{s:,}: {v:.3f}" for s, v in curve.items()) + f" steps ({passes:.0f} passes)")