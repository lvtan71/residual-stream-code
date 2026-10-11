import hashlib
import pickle

# Two prefix caches over token ids, with the same interface:
#   match(tokens)  -> how many leading tokens are cached (their KV reusable)
#   insert(tokens) -> cache them (a finished request's prompt and answer)
# capacity is in tokens; when full, least recently used entries are evicted.
#
# RadixCache is SGLang's: a radix tree whose edges hold runs of tokens, one
# token per page, so any shared prefix is found to the token. Only leaves are
# evicted, so a shared ancestor stays as long as anything below it is used.
# BlockCache is vLLM's: 16-token blocks named by a hash of the block's
# tokens and its parent's hash; only full blocks are cached, and a lookup
# stops at the first block it hasn't seen. Among blocks last used at the same
# time, the one deepest in its chain goes first, as vLLM frees a request's
# blocks in reverse order.

class Node:
    def __init__(self, parent=None, key=()):
        self.parent, self.key, self.children, self.last = parent, key, {}, 0

class RadixCache:
    def __init__(self, capacity):
        self.root, self.size, self.capacity, self.clock = Node(), 0, capacity, 0

    def _walk(self, tokens, split):
        """Follow tokens down the tree; return the deepest node reached and
        how many tokens matched. With split, cut an edge that matches only
        partly, so the match ends exactly at a node."""
        self.clock += 1
        node, i = self.root, 0
        while i < len(tokens) and tokens[i] in node.children:
            child = node.children[tokens[i]]
            k = 0
            while k < len(child.key) and i + k < len(tokens) and child.key[k] == tokens[i + k]:
                k += 1
            if k < len(child.key):
                if not split:
                    child.last = self.clock
                    return node, i + k
                mid = Node(node, child.key[:k])          # split the edge at k
                node.children[tokens[i]] = mid
                child.parent, child.key = mid, child.key[k:]
                mid.children[child.key[0]] = child
                child = mid
            child.last = self.clock
            node, i = child, i + k
        return node, i

    def match(self, tokens):
        return self._walk(tokens, split=False)[1]

    def insert(self, tokens):
        node, i = self._walk(tokens, split=True)
        if i < len(tokens):
            leaf = Node(node, tuple(tokens[i:]))
            leaf.last = self.clock
            node.children[tokens[i]] = leaf
            self.size += len(tokens) - i
        while self.size > self.capacity:
            self._evict_one()

    def _evict_one(self):
        leaves, stack = [], [self.root]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if not n.children and n is not self.root:
                leaves.append(n)
        lru = min(leaves, key=lambda n: n.last)
        del lru.parent.children[lru.key[0]]
        self.size -= len(lru.key)

class BlockCache:
    def __init__(self, capacity, block=16):
        self.blocks, self.capacity, self.block, self.clock = {}, capacity // block, block, 0

    def _hashes(self, tokens):
        parent, out = b"", []
        for i in range(0, len(tokens) - len(tokens) % self.block, self.block):
            parent = hashlib.sha256(pickle.dumps((parent, tuple(tokens[i:i + self.block])))).digest()
            out.append(parent)
        return out

    def match(self, tokens):
        self.clock += 1
        n = 0
        for depth, h in enumerate(self._hashes(tokens)):
            if h not in self.blocks:
                break
            self.blocks[h] = (self.clock, -depth)
            n += 1
        return n * self.block

    def insert(self, tokens):
        self.clock += 1
        for depth, h in enumerate(self._hashes(tokens)):
            self.blocks[h] = (self.clock, -depth)          # (last use, deeper blocks first on ties)
        while len(self.blocks) > self.capacity:
            del self.blocks[min(self.blocks, key=self.blocks.get)]
