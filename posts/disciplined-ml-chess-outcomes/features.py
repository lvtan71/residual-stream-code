"""Replay the first ten moves of every game and record what the board shows.

Reads games.parquet (from data.py), writes features.parquet: one row per game
with ratings, time control, and board features after move 3 (6 plies) and
after move 10 (20 plies). A feature is NaN when the game ended before that
point. FEN strings of the move-10 positions are kept for engine.py.
"""

import re
from multiprocessing import Pool

import chess
import numpy as np
import pandas as pd

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
MINORS = {chess.WHITE: chess.SquareSet(chess.BB_B1 | chess.BB_C1 | chess.BB_F1 | chess.BB_G1),
          chess.BLACK: chess.SquareSet(chess.BB_B8 | chess.BB_C8 | chess.BB_F8 | chess.BB_G8)}
FIRST_MOVES = ["e4", "d4", "Nf3", "c4", "e3", "g3", "b3", "f4", "d3", "Nc3"]


def material(board, color):
    return sum(v * len(board.pieces(p, color)) for p, v in VALUE.items())


def developed(board, color):
    """Minor pieces no longer on their starting squares."""
    home = sum(1 for sq in MINORS[color] if board.piece_type_at(sq) in (chess.KNIGHT, chess.BISHOP)
               and board.color_at(sq) == color)
    return 4 - home


def snapshot(board, counts):
    return {
        "material_diff": material(board, chess.WHITE) - material(board, chess.BLACK),
        "captures_w": counts["cap"][chess.WHITE], "captures_b": counts["cap"][chess.BLACK],
        "checks_w": counts["chk"][chess.WHITE], "checks_b": counts["chk"][chess.BLACK],
        "castled_w": counts["castle"][chess.WHITE], "castled_b": counts["castle"][chess.BLACK],
        "developed_w": developed(board, chess.WHITE), "developed_b": developed(board, chess.BLACK),
        "queen_moves_w": counts["queen"][chess.WHITE], "queen_moves_b": counts["queen"][chess.BLACK],
    }


def replay(movetext):
    """Board features after 6 and 20 plies, the first move, and the move-10 FEN."""
    sans = [t.rstrip("?!") for t in re.sub(r"\{[^}]*\}", "", movetext).split()
            if not re.match(r"^(\d+\.+|1-0|0-1|1/2-1/2|\*|\$\d+)$", t)]   # drop numbers, results, NAGs
    board = chess.Board()
    counts = {k: {chess.WHITE: 0, chess.BLACK: 0} for k in ("cap", "chk", "castle", "queen")}
    out = {"first_move": sans[0] if sans else None, "m3": None, "m10": None, "fen10": None}
    for ply, san in enumerate(sans[:20], start=1):
        move = board.parse_san(san)
        side = board.turn
        counts["cap"][side] += board.is_capture(move)
        counts["castle"][side] |= board.is_castling(move)
        counts["queen"][side] += board.piece_type_at(move.from_square) == chess.QUEEN
        board.push(move)
        counts["chk"][side] += board.is_check()
        if ply == 6:
            out["m3"] = snapshot(board, counts)
        if ply == 20:
            out["m10"] = snapshot(board, counts)
            out["fen10"] = board.fen()
    return out


def row(args):
    movetext, base = args
    r = replay(movetext)
    for k in ("m3", "m10"):
        for name, v in (r[k] or {}).items():
            base[f"{k}_{name}"] = v
    base["first_move"] = r["first_move"] if r["first_move"] in FIRST_MOVES else "other"
    base["fen10"] = r["fen10"]
    return base


if __name__ == "__main__":
    g = pd.read_parquet("games.parquet")
    base, inc = zip(*(tc.split("+") for tc in g.TimeControl))
    meta = pd.DataFrame({
        "game": g.Site.str.rsplit("/", n=1).str[-1],
        "score": g.score, "termination": g.Termination,
        "elo_w": g.WhiteElo.astype(int), "elo_b": g.BlackElo.astype(int),
        "tc_base": np.array(base, dtype=int), "tc_inc": np.array(inc, dtype=int),
    })
    with Pool(4) as pool:
        rows = pool.map(row, zip(g.movetext, meta.to_dict("records")), chunksize=500)
    f = pd.DataFrame(rows)
    f["elo_diff"] = f.elo_w - f.elo_b
    f["elo_mean"] = (f.elo_w + f.elo_b) / 2
    f.to_parquet("features.parquet")
    print(f"{len(f):,} games, {f.shape[1]} columns")
    print(f"move-3 features present: {f.m3_material_diff.notna().mean():.1%}; "
          f"move-10: {f.m10_material_diff.notna().mean():.1%}")
    print(f"first moves: " + ", ".join(f"{k} {v:.1%}" for k, v in
                                       f.first_move.value_counts(normalize=True).head(5).items()))
