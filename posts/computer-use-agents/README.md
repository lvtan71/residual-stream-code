# Computer-Use Agents: the full code

The scripts behind [Computer-Use Agents: Screens as Observations, the Mouse as a Tool](https://residual-stream.lvtan71.workers.dev/posts/computer-use-agents/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `observation_cost.py` | Screenshot tokens at four screen sizes on Claude's standard and high-resolution tiers, from the documented 28×28-patch rule; tokens of three pages' accessibility trees, whole and pruned to interactive elements; and one task's input tokens and cost over 10, 30 and 50 steps, keeping every screenshot or the last 3, with and without prompt caching. | CPU, under a minute; downloads Qwen2.5-1.5B-Instruct's tokenizer files (a few MB) at a pinned revision |

The accessibility trees were snapshots of three live pages (this blog's home page, Hacker News and Wikipedia), captured once with Chrome's devtools as `a11y_blog_home.txt`, `a11y_hackernews.txt` and `a11y_wikipedia.txt`. The files were not kept, and the pages have changed since, so the script skips those three lines of its output when the files are missing. A new snapshot saved under the same name is counted the same way, with different numbers.

The outputs here came from an Apple M4 laptop with transformers 5.17. The huggingface_hub version was not recorded, so it is left unpinned.

```bash
pip install "transformers==5.17.*" huggingface_hub
python observation_cost.py
```
