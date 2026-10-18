"""The research agent the post's experiments run: a synthetic corpus with
planted facts, three ordinary tools, an instrumented agent loop and a
client-side memory backend. The experiment scripts import it.

Needs ANTHROPIC_API_KEY in the environment.
"""
import json, os, shutil, tempfile
from dataclasses import dataclass, field
from pathlib import Path
import anthropic

MODEL = os.environ.get("MODEL", "claude-sonnet-5-5")   # the post's runs; MODEL=claude-haiku-5-5 costs far less
client = anthropic.Anthropic()

N_DOCS, DOC_TOKENS, MAX_TURNS = 6, 12_000, 12
BETA_CLEAR = "context-management-2025-06-27"
BETA_COMPACT = "compact-2026-01-12"

# The corpus: each review is filler plus two planted facts, a headline fact
# (the lifespan) and an obscure one (an appendix provenance string), so a
# summary can be probed for what it kept.
ORGANISMS = [
    ("celegans",   "C. elegans",      "18 days",      "A-7", "cohort jar #414"),
    ("drosophila", "D. melanogaster", "60 days",      "A-3", "vial rack #221"),
    ("yeast",      "S. cerevisiae",   "25 divisions", "A-9", "plate lot #877"),
    ("killifish",  "N. furzeri",      "16 weeks",     "A-2", "tank block #305"),
    ("mouse",      "M. musculus",     "30 months",    "A-5", "cage bank #692"),
    ("zebrafish",  "D. rerio",        "42 months",    "A-4", "raceway #158"),
][:N_DOCS]
FILLER = ("Longitudinal survival analysis across replicate cohorts indicates that "
          "environmental variance contributes materially to observed dispersion, and "
          "that husbandry protocol standardization reduces between-lab heterogeneity. ")
def dense_text(rng, name, n_chars):
    """Discussion made of distinct cohort measurements, each a specific a
    summary could keep or drop."""
    out = []
    while sum(map(len, out)) < n_chars:
        out.append(f"In cohort {rng.choice('BCDEFGH')}-{rng.randint(10, 99)}, median survival of {name} "
                   f"was {rng.randint(40, 160)}% of controls (n={rng.randint(20, 400)}, "
                   f"{rng.choice(['dietary restriction', 'rapamycin', 'metformin', 'heat stress', 'daf-2 RNAi'])}, "
                   f"lab {rng.randint(1, 40)}). ")
    return "".join(out)

def make_corpus(dense=False, seed=0):
    """The plain corpus pads each review with one repeated sentence; the dense
    one fills it with distinct measurements and hides the planted source among
    30 lookalike provenance rows."""
    import random
    rng = random.Random(seed)
    corpus = {}
    for slug, name, lifespan, table, source in ORGANISMS:
        body = (dense_text(rng, name, DOC_TOKENS * 4) if dense
                else FILLER * (DOC_TOKENS * 4 // len(FILLER)))
        kind = source.rsplit(" #", 1)[0]
        rows = [f"{kind} #{rng.randint(100, 999)}" for _ in range(29)]
        rows.insert(rng.randint(0, len(rows)), source)
        corpus[f"/research/{slug}_review.md"] = (
            f"# {name} as a model organism for aging research\n\n## Headline\n"
            f"The median lifespan of {name} under standard laboratory conditions is **{lifespan}**.\n\n"
            f"## Discussion\n{body}\n\n## Appendix table {table}\n" +
            ("".join(f"Supplementary provenance: the {name} replicate cohort {'primary' if r == source else f'R{i}'} "
                     f"was sourced from {r}.\n" for i, r in enumerate(rows)) if dense else
             f"Supplementary provenance: the {name} replicate cohort was sourced from {source}.\n"))
    return corpus

CORPUS = make_corpus()
FACTS = {name: lifespan for _, name, lifespan, *_ in ORGANISMS}
OBSCURE = {name: source for _, name, _, _, source in ORGANISMS}

TOOLS = [
    {"name": "search_files", "description": "Search document filenames and contents for a keyword.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "read_file", "description": "Read the full content of a document by path.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "record_finding", "description": "Record a finding. Session-local; not persisted across sessions.",
     "input_schema": {"type": "object", "properties": {"finding": {"type": "string"}}, "required": ["finding"]}},
]
MEMORY_TOOL = {"type": "memory_20250818", "name": "memory"}
SYSTEM = ("You are a biology research analyst comparing model organisms for aging research. "
          "Work systematically: read the review documents, record notes on key facts, and "
          "build a comparative picture. Be concise in your reasoning text; the goal is notes, "
          "not essays.")
TASK = ("Compare the model organisms in /research/ on lifespan and experimental throughput. "
        "Read every review document, record one finding per organism, then write a short "
        "comparative synthesis.")

USAGE = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
_counts = {}

def count_tokens(text):
    """Exact token count from the API's counting endpoint (free), memoized."""
    if text not in _counts:
        _counts[text] = client.messages.count_tokens(
            model=MODEL, messages=[{"role": "user", "content": text}]).input_tokens
    return _counts[text]

def run_tool(name, args, notes):
    if name == "search_files":
        q = (args.get("query") or "").lower()
        hits = [p for p, c in CORPUS.items() if q in p.lower() or q in c.lower()]
        return f"Found {len(hits)}:\n" + "\n".join(hits) if hits else f"No matches for '{q}'."
    if name == "read_file":
        path = args.get("path", "")
        return CORPUS.get(path, f"Error: '{path}' not found. Available: {', '.join(CORPUS)}")
    if name == "record_finding":
        notes.append(args.get("finding", ""))
        return f"Finding #{len(notes)} recorded (session-local)."
    return f"Error: unknown tool '{name}'"

@dataclass
class Session:
    trajectory: list = field(default_factory=list)   # (turn, context tokens)
    events: list = field(default_factory=list)       # clearing and compaction events
    reads: list = field(default_factory=list)        # (turn, path) of every read_file call
    notes: list = field(default_factory=list)
    summaries: list = field(default_factory=list)    # compaction summaries' text
    finished: bool = False                           # gave a final answer within MAX_TURNS

    @property
    def peak(self):   # the largest context sent, counting what a compaction replaced
        return max([t for _, t in self.trajectory] + [e["before"] for e in self.events if "before" in e] or [0])

def context_size(u):
    # input_tokens is only the uncached remainder; the context is all three
    return u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)

def run_session(prompt, *, edits=None, memory=None, label="session", max_tokens=4096):
    """The agent loop. `edits` is the context_management edits list; `memory`
    a MemoryBackend to expose the memory tool with. A compaction's summary is
    written within the request's max_tokens."""
    tools = TOOLS + ([MEMORY_TOOL] if memory else [])
    messages, s = [{"role": "user", "content": prompt}], Session()
    betas = sorted({BETA_CLEAR if e["type"].startswith("clear") else BETA_COMPACT for e in edits or []})
    print(f"[{label}]")
    for turn in range(1, MAX_TURNS + 1):
        kw = dict(model=MODEL, max_tokens=max_tokens, system=SYSTEM, tools=tools, messages=messages)
        if edits:
            kw.update(betas=betas, context_management={"edits": edits})
        resp = client.beta.messages.create(**kw)
        u = resp.usage
        # the top-level counts leave out a compaction's own summarizing call;
        # usage.iterations lists every sampling step the request billed
        for it in getattr(u, "iterations", None) or [u]:
            if getattr(it, "type", "") == "compaction":   # the context the summary replaced
                s.events.append({"turn": turn, "kind": "compaction", "before": it.input_tokens,
                                 "summary_out": it.output_tokens})
            bill({"input": it.input_tokens, "cache_write": getattr(it, "cache_creation_input_tokens", 0),
                  "cache_read": getattr(it, "cache_read_input_tokens", 0), "output": it.output_tokens})
        s.trajectory.append((turn, context_size(u)))
        cm = getattr(resp, "context_management", None)
        for e in getattr(cm, "applied_edits", None) or []:
            if "clear" in e.type:
                s.events.append({"turn": turn, "kind": "clearing", "results": e.cleared_tool_uses,
                                 "tokens": e.cleared_input_tokens})
        content, calls = [], []
        for b in resp.content:
            if b.type == "compaction":
                s.summaries.append(b.content or "")   # None if the summarizing step failed
            elif b.type == "tool_use":
                calls.append(b)
                if b.name == "read_file":
                    s.reads.append((turn, b.input.get("path", "?")))
            content.append(b.model_dump(exclude_none=True))
        messages.append({"role": "assistant", "content": content})   # all blocks, compaction included
        tag = " ".join(f"{e['kind']}" for e in s.events if e["turn"] == turn)
        print(f"  turn {turn:2d}  context {s.trajectory[-1][1]:>7,}  "
              f"{', '.join(c.name for c in calls) or 'final answer'}  {tag}".rstrip())
        if not calls:
            s.finished = True
            break
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id,
             "content": memory.handle(c.input) if c.name == "memory" else run_tool(c.name, c.input, s.notes)}
            for c in calls]})
    return s

def probe(summary, names):
    """Ask the model, given only the summary, for each organism's lifespan and
    the identifier its primary cohort was sourced from: what an agent
    continuing from the summary could still answer. Returns, per organism,
    whether each answer matches the planted fact."""
    import re
    ask = "\n".join(f"- {n}: median lifespan? the identifier its replicate cohort (the primary one, if several "
                    f"are listed) was sourced from?" for n in names)
    reply = client.messages.create(model=MODEL, max_tokens=1024, messages=[{"role": "user", "content":
        f"<summary>\n{summary}\n</summary>\n\nUsing only the summary above, answer for each organism. "
        f"Reply with one line per organism in the form `name | lifespan | identifier`, writing "
        f"`unknown` for anything the summary does not state.\n{ask}"}])
    bill({"input": reply.usage.input_tokens, "output": reply.usage.output_tokens})
    lines = {l.split("|")[0].strip(" -`*"): l.split("|")[1:] for l in "".join(b.text for b in reply.content if b.type == "text").splitlines() if l.count("|") >= 2}
    def num(x):
        return re.findall(r"\d+", x)
    out = {}
    for n in names:
        life, ident = (lines.get(n) or ["", ""])[:2]
        out[n] = {"headline": num(FACTS[n]) == num(life)[:1] and FACTS[n].split()[1][:3] in life.lower(),
                  "obscure": re.findall(r"\d{3}", OBSCURE[n]) == re.findall(r"\d{3}", ident)[:1]}
    out["reply"] = "".join(b.text for b in reply.content if b.type == "text")
    return out

class MemoryBackend:
    """Client-side storage for memory_20250818: the API defines the commands,
    the application stores the files."""
    def __init__(self, base=None):
        self.base = Path(base or tempfile.mkdtemp(prefix="memories_"))
        self.base.mkdir(parents=True, exist_ok=True)

    def resolve(self, path):
        # model-written paths must stay inside the memory root
        rel = path.removeprefix("/memories").removeprefix("/") or "."
        p = (self.base / rel).resolve()
        if not p.is_relative_to(self.base.resolve()):
            raise ValueError(f"path escapes the memory directory: {path}")
        return p

    def handle(self, a):
        try:
            cmd = a.get("command")
            if cmd == "view":
                p = self.resolve(a.get("path", "/memories"))
                if p.is_dir():
                    return "\n".join(sorted(e.name + "/" * e.is_dir() for e in p.iterdir())) or "(empty)"
                return "\n".join(f"{i + 1:4d}\t{l}" for i, l in enumerate(p.read_text().split("\n")))
            if cmd == "create":
                p = self.resolve(a["path"]); p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(a["file_text"]); return f"Created {a['path']}"
            if cmd == "str_replace":
                p = self.resolve(a["path"]); t = p.read_text()
                if t.count(a["old_str"]) != 1:
                    return f"Error: old_str appears {t.count(a['old_str'])} times in {a['path']}"
                p.write_text(t.replace(a["old_str"], a["new_str"])); return f"Replaced in {a['path']}"
            if cmd == "insert":
                p = self.resolve(a["path"]); lines = p.read_text().split("\n")
                lines.insert(a["insert_line"], a["insert_text"]); p.write_text("\n".join(lines))
                return f"Inserted at line {a['insert_line']} in {a['path']}"
            if cmd == "delete":
                p = self.resolve(a["path"])
                shutil.rmtree(p) if p.is_dir() else p.unlink(); return f"Deleted {a['path']}"
            if cmd == "rename":
                o, n = self.resolve(a["old_path"]), self.resolve(a["new_path"])
                n.parent.mkdir(parents=True, exist_ok=True); o.rename(n)
                return f"Renamed {a['old_path']} to {a['new_path']}"
            return f"Error: unknown command '{cmd}'"
        except Exception as e:
            return f"Error: {e}"

    def files(self):
        return {str(p.relative_to(self.base)): p.read_text() for p in sorted(self.base.rglob("*")) if p.is_file()}

def save(name, s, **extra):
    """Write a run's measurements to results/<name>.json for the figures."""
    os.makedirs("results", exist_ok=True)
    json.dump({"trajectory": s.trajectory, "events": s.events, "reads": s.reads, "peak": s.peak,
               "summaries": s.summaries, **extra}, open(f"results/{name}.json", "w"), indent=1)

# $ per million tokens, from platform.claude.com/docs/en/about-claude/pricing
# (read 2026-10-18). Haiku 5.5 charges more for a request whose prompt, all
# its input whether cached or not, is over 100,000 tokens.
PRICES = {"claude-sonnet-5-5": [(None, {"input": 2.00, "cache_write": 2.50, "cache_read": 0.10, "output": 10.00})],
          "claude-haiku-5-5": [(100_000, {"input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50}),
                               (None, {"input": 0.50, "cache_write": 0.625, "cache_read": 0.05, "output": 2.50})]}
COST = [0.0]

def bill(counts):
    """Add one billed request's token counts to USAGE and its price to COST."""
    counts = {k: counts.get(k) or 0 for k in USAGE}
    prompt = counts["input"] + counts["cache_write"] + counts["cache_read"]
    price = next(p for limit, p in PRICES[MODEL] if limit is None or prompt <= limit)
    for k in USAGE:
        USAGE[k] += counts[k]
    COST[0] += sum(counts[k] * price[k] for k in USAGE) / 1e6

def print_usage():
    cost = COST[0]
    print(f"API usage: {USAGE['input']:,} uncached input, {USAGE['cache_write']:,} cache-write, "
          f"{USAGE['cache_read']:,} cache-read, {USAGE['output']:,} output tokens; ${cost:.2f}")
