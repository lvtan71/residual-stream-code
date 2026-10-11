import hashlib
import pickle
from radix import BlockCache

# TensorRT-LLM's default prefix cache: blocks of 32 tokens, named by a hash
# of their tokens and their parent's, with partial reuse on
# (enable_partial_reuse and copy_on_partial_reuse default to True). Whole
# blocks are matched by hash; the first block that doesn't match whole is
# compared token by token with the blocks stored after the last match, and
# the longest common run is reused (copied into a new block). A request's
# last, partly filled block is stored too. Eviction is radix.py's: least
# recently used, the deepest block first among ties.

class PartialBlockCache(BlockCache):
    def __init__(self, capacity, block=32):
        super().__init__(capacity, block)
        self.children, self.parent = {}, {}      # parent hash -> {hash: tokens}; hash -> parent hash

    def _chain(self, tokens):
        parent, out = b"", []
        for i in range(0, len(tokens), self.block):
            run = tuple(tokens[i:i + self.block])
            h = hashlib.sha256(pickle.dumps((parent, run))).digest()
            out.append((parent, h, run))
            parent = h
        return out

    def match(self, tokens):
        self.clock += 1
        n = 0
        for depth, (parent, h, run) in enumerate(self._chain(tokens)):
            if len(run) == self.block and h in self.blocks:      # a whole block, by hash
                self.blocks[h] = (self.clock, -depth)
                n += self.block
                continue
            best, hit = 0, None                                  # partial: compare token by token
            for child, stored in self.children.get(parent, {}).items():
                k = 0
                while k < min(len(run), len(stored)) and run[k] == stored[k]:
                    k += 1
                if k > best:
                    best, hit = k, child
            if hit is not None:
                self.blocks[hit] = (self.clock, -depth)
                n += best
            break
        return n

    def insert(self, tokens):
        self.clock += 1
        for depth, (parent, h, run) in enumerate(self._chain(tokens)):
            self.blocks[h] = (self.clock, -depth)
            self.children.setdefault(parent, {})[h] = run
            self.parent[h] = parent
        while len(self.blocks) > self.capacity:
            h = min(self.blocks, key=self.blocks.get)
            del self.blocks[h], self.children[self.parent.pop(h)][h]
