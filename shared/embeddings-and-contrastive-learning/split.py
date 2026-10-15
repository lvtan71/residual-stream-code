import re
import torch

# Split the blog into sections of raw words once: 30% of sections are held
# out for testing and never trained on, not even by word2vec. Everything
# else, sections and the text between them, is training text.
VIEW = 40                                                      # words per view
text = open("blog_train.txt").read() + open("blog_val.txt").read()
chunks = re.split(r"\n## ", text)
long_enough = [i for i, c in enumerate(chunks) if len(c.split()) >= 3 * VIEW]
order = torch.randperm(len(long_enough), generator=torch.Generator().manual_seed(0))
held_out = {long_enough[i] for i in order[:len(long_enough) * 3 // 10]}
test_secs = [chunks[i].split() for i in sorted(held_out)]
train_secs = [chunks[i].split() for i in long_enough if i not in held_out]
train_text = "\n## ".join(c for i, c in enumerate(chunks) if i not in held_out)
