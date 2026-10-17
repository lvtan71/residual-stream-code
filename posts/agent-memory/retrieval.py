import numpy as np
from sentence_transformers import SentenceTransformer

# Agent memory, measured without a language model in the loop: does a
# memory policy put the right fact in front of the model? A synthetic
# 50-turn conversation per trial: 8 facts about the user, mixed into small
# talk; 3 of the facts are later updated ("I moved", "I switched"). At the
# end, one question per fact. A policy may show the model k memories, and
# each question is classified by what it shows: the current version only,
# both versions (the model must work out which is newer), the outdated
# version only, or neither. 50 trials, with facts and filler drawn at random.
# Importance would be rated by a model (Park et al. ask it for 1 to 10);
# here facts are given 8 and small talk 3, a perfect rater, so two
# ablations drop one term each. 5 seeds x 50 conversations.
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", device="cpu")
FACTS = [  # (subject, statement template, question, old values, new values)
    ("city", "I live in {}.", "Which city does the user live in?", ["Da Nang", "Hue", "Can Tho"], ["Hanoi", "Saigon", "Hai Phong"]),
    ("job", "I work as a {}.", "What is the user's job?", ["teacher", "nurse", "designer"], ["data engineer", "pilot", "chef"]),
    ("pet", "My pet is a {}.", "What pet does the user have?", ["kitten", "parrot", "rabbit"], ["dog", "turtle", "hamster"]),
    ("language", "I'm learning {}.", "Which language is the user learning?", ["Japanese", "French", "Korean"], ["German", "Spanish", "Thai"]),
    ("diet", "I'm {}.", "What is the user's diet?", ["vegetarian", "vegan", "pescatarian"], ["on a keto diet", "eating everything", "gluten-free"]),
    ("sport", "I play {} on weekends.", "Which sport does the user play?", ["badminton", "tennis", "football"], ["basketball", "volleyball", "chess"]),
    ("car", "I drive a {}.", "What car does the user drive?", ["Toyota", "Honda", "Kia"], ["VinFast", "Tesla", "Mazda"]),
    ("music", "My favorite band is {}.", "What is the user's favorite band?", ["Coldplay", "BTS", "Queen"], ["Radiohead", "Muse", "Oasis"]),
]
UPDATE = "Actually, things changed: {}"
FILLER = ["The weather is nice today.", "I had coffee this morning.", "Traffic was bad.", "I watched a movie last night.",
          "Work was busy this week.", "I need to buy groceries.", "My phone battery is low.", "I'm a bit tired.",
          "That article was interesting.", "Let's talk about something else.", "I went for a walk.", "Lunch was good."]

def trial(rng):
    turns = [rng.choice(FILLER) for _ in range(50)]
    slots = rng.choice(np.arange(2, 35), 8, replace=False)
    updated = set(rng.choice(8, 3, replace=False))
    truth = {}
    for f, (name, tpl, q, old, new) in enumerate(FACTS):
        v_old, v_new = rng.choice(old), rng.choice(new)
        turns[slots[f]] = tpl.format(v_old)
        truth[f] = (v_old, v_new if f in updated else None)
        if f in updated:                                            # the update comes later, replacing small talk
            free = [i for i in range(slots[f] + 5, 50) if turns[i] in FILLER and i not in slots]
            turns[int(rng.choice(free))] = UPDATE.format(tpl.format(v_new))
    importance = np.array([3.0 if t in FILLER else 8.0 for t in turns])   # a model would rate this; facts rate high
    return turns, truth, importance

def shown(policy, turns, importance, E, q, k):
    now = len(turns)
    rel = E @ q
    if policy == "last k turns":
        return list(range(now - k, now))
    if policy == "most relevant k":
        return list(np.argsort(-rel)[:k])
    recency = 0.99 ** (now - 1 - np.arange(now))                     # Generative Agents' three scores
    norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    use = {"recency + importance + relevance": (1, 1), "recency + relevance": (1, 0),
           "importance + relevance": (0, 1)}[policy]
    score = use[0] * norm(recency) + use[1] * norm(importance) + norm(rel)
    return list(np.argsort(-score)[:k])

POLICIES = ("last k turns", "most relevant k", "recency + importance + relevance", "recency + relevance",
            "importance + relevance")
Q = enc.encode([q for _, _, q, _, _ in FACTS], normalize_embeddings=True)
count = {(p, k): np.zeros(4) for p in POLICIES for k in (1, 3, 5)}   # current only, both, outdated only, neither
for seed in range(5):
  rng = np.random.default_rng(seed)
  for _ in range(50):
    turns, truth, importance = trial(rng)
    E = enc.encode(turns, normalize_embeddings=True)
    for f in range(len(FACTS)):
        v_old, v_new = truth[f]
        for p in POLICIES:
            for k in (1, 3, 5):
                text = " ".join(turns[i] for i in shown(p, turns, importance, E, Q[f], k))
                cur = (v_new or v_old) in text
                old = v_new is not None and v_old in text
                count[(p, k)][0 if cur and not old else 1 if cur and old else 2 if old else 3] += 1
print(f"{'':34s} {'k':>2s} {'current only':>13s} {'both':>6s} {'outdated only':>14s} {'neither':>8s}")
for p in POLICIES:
    for k in ((1, 3, 5) if p in POLICIES[:3] else (1,)):
        c = count[(p, k)] / count[(p, k)].sum()
        print(f"{p:34s} {k:2d} {c[0]:13.0%} {c[1]:6.0%} {c[2]:14.0%} {c[3]:8.0%}")

# Worked: one changed fact (from the last conversation) where relevance
# alone prefers the outdated statement, and the three scores for its top
# three memories.
norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
now = len(turns)
recency = 0.99 ** (now - 1 - np.arange(now))
for f in range(len(FACTS)):
    if not truth[f][1]:
        continue
    rel = E @ Q[f]
    orig = turns.index(FACTS[f][1].format(truth[f][0]))
    upd = turns.index(UPDATE.format(FACTS[f][1].format(truth[f][1])))
    if rel[orig] > rel[upd]:
        break
total = norm(recency) + norm(importance) + norm(rel)
print(f'\nworked: "{FACTS[f][2]}"')
for i in np.argsort(-total)[:3]:
    print(f"  turn {i:2d} {turns[i][:52]!r:56s} recency {norm(recency)[i]:.2f} + importance "
          f"{norm(importance)[i]:.2f} + relevance {norm(rel)[i]:.2f} = {total[i]:.2f}")