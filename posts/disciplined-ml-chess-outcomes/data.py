"""Download January 2015 from the Lichess database and keep the first 100,000
rated blitz games, then audit them before any model sees them.

Source: the Lichess/standard-chess-games dataset on Hugging Face, a parquet
copy of database.lichess.org, pinned to a revision. Writes games.parquet.
"""

import pandas as pd
from huggingface_hub import hf_hub_download

REPO, REV = "Lichess/standard-chess-games", "de4e636eddf568a9394cc01fb0b9e1da04a6babf"
FILES = [f"data/year=2015/month=01/train-0000{i}-of-00003.parquet" for i in range(3)]
COLUMNS = ["Event", "Site", "Result", "WhiteElo", "BlackElo", "UTCDate", "UTCTime",
           "Termination", "TimeControl", "movetext"]
N = 100_000

month = pd.concat([pd.read_parquet(hf_hub_download(REPO, f, repo_type="dataset", revision=REV),
                                   columns=COLUMNS) for f in FILES], ignore_index=True)
blitz = month[month.Event.str.startswith("Rated Blitz")]
games = blitz.iloc[:N].reset_index(drop=True)       # file order, fixed before looking at results
games["score"] = games.Result.map({"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0})
games["moves"] = games.movetext.str.count(r"\d+\. ")  # full moves started
games.to_parquet("games.parquet")

print(f"January 2015: {len(month):,} rated games, {len(blitz):,} blitz; kept the first {len(games):,}")
when = pd.to_datetime(games.UTCDate.astype(str) + " " + games.UTCTime.astype(str))
span = (when.max() - when.min()).total_seconds() / 86400
print(f"they span {when.min()} to {when.max()} UTC: {span:.1f} of the month's 31 days")
print(f"results: White wins {(games.score == 1).mean():.1%}, draws {(games.score == 0.5).mean():.1%}, "
      f"Black wins {(games.score == 0).mean():.1%}; White's mean score {games.score.mean():.3f}")
print(f"ratings missing: {games.WhiteElo.isna().sum() + games.BlackElo.isna().sum()}")

print("\nhow games ended, and how many moves they lasted")
print(f"{'termination':>16s} {'games':>7s} {'median moves':>13s} {'under 3 moves':>14s}")
for term, g in games.groupby("Termination"):
    print(f"{term:>16s} {len(g):>7,} {g.moves.median():>13.0f} {(g.moves < 3).mean():>14.1%}")
ab = games[games.Termination == "Abandoned"]
print(f"abandoned games: White's mean score {ab.score.mean():.3f}, decisive {(ab.score != 0.5).mean():.1%}, "
      f"moves played {ab.moves.sum()}")
for k in (3, 10):
    print(f"games reaching move {k}: {(games.moves >= k).mean():.1%}")
