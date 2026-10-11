import torch
import torch.nn.functional as F
from lm import model, tok, DEVICE

# The residual stream of a real model, Qwen2.5-1.5B-Instruct (28 blocks,
# stream width 1,536), on 512 tokens of this blog. Hooks read the stream
# entering each block, and the attention and MLP updates each block adds.
# For every point along the stream: its size (root mean square of the
# entries), each sublayer's update against the stream it reads (attention
# reads the block's input; the MLP, that plus attention's update), and what
# it would predict if the model stopped there, read
# through the final norm and the unembedding (the "logit lens"): the
# next-token loss, and how often its top token is the final prediction.
# Then each block is removed in turn (its update skipped) to see what it
# was worth. The first position is left out of the sizes: its stream grows
# far larger than the others' (printed separately); the losses include it.
ids = torch.tensor([tok(open("blog_train.txt").read()).input_ids[1000:1512]], device=DEVICE)
blocks = model.model.layers
streams, mids, updates, skip = [], [], {"attn": [], "mlp": []}, None

record = True                                     # off during the removal runs

def save_input(module, args, kwargs):
    if record:
        streams.append((args[0] if args else kwargs["hidden_states"]).float())
def save_update(kind):
    def hook(module, args, out):
        if record:
            updates[kind].append((out[0] if isinstance(out, tuple) else out).float())
    return hook
def skip_block(i):
    def hook(module, args, kwargs, out):
        if i == skip:
            return args[0] if args else kwargs["hidden_states"]          # the block adds nothing
    return hook

for i, b in enumerate(blocks):
    b.register_forward_pre_hook(save_input, with_kwargs=True)
    b.self_attn.register_forward_hook(save_update("attn"))
    b.mlp.register_forward_hook(save_update("mlp"))
    b.post_attention_layernorm.register_forward_pre_hook(          # the stream the MLP reads
        lambda m, args: mids.append(args[0].float()) if record else None)
    b.register_forward_hook(skip_block(i), with_kwargs=True)
model.model.norm.register_forward_pre_hook(lambda m, args: streams.append(args[0].float()) if record else None)

lens = lambda x: model.lm_head(model.model.norm(x.to(model.dtype))).float()
loss = lambda logits: F.cross_entropy(logits[0, :-1], ids[0, 1:]).item()
rms = lambda x: x[0, 1:].pow(2).mean(-1).sqrt().mean().item()             # positions 1 onward

with torch.no_grad():
    final = model(ids).logits.float()
    record = False                                # the lens below calls the final norm again
    pred = final[0, 1:].argmax(-1)
    print(f"512 tokens; final next-token loss {loss(final):.3f} nats; stream width {streams[0].shape[-1]}")
    print("                                    the next block adds, relative to")
    print("                                    the stream each sublayer reads:")
    print("point in the stream     size (RMS)   attention        MLP        logit-lens loss   top token = final")
    for l, x in enumerate(streams):
        name = "embedding" if l == 0 else f"after block {l}"
        upd = ""
        if l < len(blocks):
            upd = f"{rms(updates['attn'][l]) / rms(x):12.2f}x  {rms(updates['mlp'][l]) / rms(mids[l]):11.2f}x"
        else:
            upd = " " * 27
        agree = (lens(x)[0, 1:].argmax(-1) == pred).float().mean().item()
        print(f"{name:20s}  {rms(x):9.2f}   {upd}   {loss(lens(x)):13.2f}   {agree:15.0%}")
    first = [x[0, 0].pow(2).mean().sqrt().item() for x in streams]
    print(f"first position's stream size: {first[0]:.2f} at the embedding, up to {max(first):.0f} "
          f"(after block {first.index(max(first))}); others' at most {max(rms(x) for x in streams):.0f}")
    print("block removed:  loss  (increase)")
    for i in range(len(blocks)):
        skip = i
        lost = loss(model(ids).logits.float())
        print(f"  block {i + 1:2d}       {lost:.3f}  (+{lost - loss(final):.3f})")
    skip = None
