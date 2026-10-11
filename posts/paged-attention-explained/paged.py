import torch
from transformers.cache_utils import Cache, DynamicLayer

# A paged KV cache for a Hugging Face model, as vLLM lays it out: one pool
# of fixed-size physical blocks per layer, each holding the keys and values
# of BLOCK tokens, and per sequence a block table, the list of physical
# blocks that hold its tokens in order. The same table serves every layer.
# Slots are allocated before each forward pass, as vLLM's scheduler does;
# the layers then write the new keys and values into those slots, and
# gather the sequence's blocks back into order to attend over them.
BLOCK = 16

class Pool:
    def __init__(self, n_blocks, layers, kv_heads, head_dim, dtype, device):
        shape = (layers, n_blocks, kv_heads, BLOCK, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.free = list(range(n_blocks))            # physical blocks not in use
        self.refs = [0] * n_blocks                   # how many block tables point at each block
        self.copies = 0

    def alloc(self):
        b = self.free.pop(0)
        self.refs[b] = 1
        return b

    def release(self, b):
        self.refs[b] -= 1
        if self.refs[b] == 0:
            self.free.append(b)

class Sequence:
    def __init__(self, pool):
        self.pool, self.table, self.n = pool, [], 0   # n: tokens whose keys and values are stored

    def reserve(self, new):
        """Make room for `new` more tokens: copy a shared last block before
        writing into it (copy-on-write), then add blocks as needed."""
        p = self.pool
        if self.n % BLOCK and p.refs[self.table[-1]] > 1:
            old, b = self.table[-1], p.alloc()
            p.k[:, b], p.v[:, b] = p.k[:, old], p.v[:, old]
            p.release(old)
            self.table[-1] = b
            p.copies += 1
        while len(self.table) * BLOCK < self.n + new:
            self.table.append(p.alloc())

    def fork(self):
        """A second sequence sharing every block of this one."""
        child = Sequence(self.pool)
        child.table, child.n = list(self.table), self.n
        for b in self.table:
            self.pool.refs[b] += 1
        return child

    def free(self):
        for b in self.table:
            self.pool.release(b)
        self.table = []

class PagedLayer(DynamicLayer):
    def __init__(self, seq, layer):
        super().__init__()
        self.seq, self.layer, self.is_initialized = seq, layer, True

    def update(self, k, v, *args, **kwargs):          # k, v: [1, kv_heads, new tokens, head_dim]
        s, p = self.seq, self.seq.pool
        pos = torch.arange(s.n, s.n + k.shape[2])
        table = torch.tensor(s.table)
        blocks, offsets = table[pos // BLOCK], pos % BLOCK
        p.k[self.layer, blocks, :, offsets] = k[0].transpose(0, 1)
        p.v[self.layer, blocks, :, offsets] = v[0].transpose(0, 1)
        total = s.n + k.shape[2]                      # gather this sequence's blocks, in order
        used = table[: -(-total // BLOCK)]
        K = p.k[self.layer, used].transpose(0, 1).flatten(1, 2)[:, :total]
        V = p.v[self.layer, used].transpose(0, 1).flatten(1, 2)[:, :total]
        return K[None], V[None]

    def get_seq_length(self):
        return self.seq.n

def paged_cache(seq, layers):
    return Cache(layers=[PagedLayer(seq, i) for i in range(layers)])

@torch.no_grad()
def step(model, seq, ids):
    """One forward pass over `ids` for `seq`: reserve slots, run, commit."""
    seq.reserve(len(ids))
    out = model(ids[None], past_key_values=paged_cache(seq, model.config.num_hidden_layers),
                use_cache=True)
    seq.n += len(ids)
    return out.logits[0, -1]
