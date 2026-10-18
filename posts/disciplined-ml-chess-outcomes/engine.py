"""Ask Stockfish who stands better after move 10, for every game that got there.

Reads features.parquet (fen10), writes engine.parquet: the evaluation in
centipawns from White's side. Needs a Stockfish binary on PATH (the outputs
used Stockfish 18). A fixed node budget and one thread per process make each
evaluation deterministic for a given binary.
"""

import shutil
import sys
from multiprocessing import Pool

import chess
import chess.engine
import pandas as pd

if shutil.which("stockfish") is None:
    sys.exit("engine.py needs a Stockfish binary on PATH; engine.parquet is committed, so engine_ab.py runs without it")

NODES = 20_000
CAP = 1_500                       # centipawns; a forced mate counts as a decisive +-1500


def evaluate(fens):
    engine = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish"))
    engine.configure({"Threads": 1, "Hash": 16})
    out = []
    for fen in fens:
        engine.protocol.send_line("ucinewgame")   # no hash carried over between games
        info = engine.analyse(chess.Board(fen), chess.engine.Limit(nodes=NODES))
        out.append(max(-CAP, min(CAP, info["score"].white().score(mate_score=100_000))))
    engine.quit()
    return out


if __name__ == "__main__":
    f = pd.read_parquet("features.parquet", columns=["game", "fen10"]).dropna()
    chunks = [f.fen10.iloc[i:i + 1000].tolist() for i in range(0, len(f), 1000)]
    with Pool(4) as pool:
        evals = [e for chunk in pool.map(evaluate, chunks) for e in chunk]
    out = pd.DataFrame({"game": f.game, "m10_engine": evals})
    out.to_parquet("engine.parquet")
    with chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish")) as e:
        print(f"{e.id['name']}, {NODES:,} nodes per position, {len(out):,} positions")
    print(f"evaluation after move 10, White's view: median {out.m10_engine.median():+.0f} cp, "
          f"|eval| > 200 cp in {(out.m10_engine.abs() > 200).mean():.1%}")
