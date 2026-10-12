"""The leakage contract: which columns each model may see, and on which games.

Every model reads its inputs from FEATURES and nothing else, so the rules
below are the only place information can enter a model.
"""

TIME = ["tc_base", "tc_inc"]
RATINGS = ["elo_w", "elo_b", "elo_diff", "elo_mean"]
BOARD = ["material_diff", "captures_w", "captures_b", "checks_w", "checks_b", "castled_w",
         "castled_b", "developed_w", "developed_b", "queen_moves_w", "queen_moves_b"]
M3 = ["first_move"] + [f"m3_{c}" for c in BOARD]
M10 = M3 + [f"m10_{c}" for c in BOARD]

FEATURES = {
    "win_pregame": RATINGS + TIME,
    "win_move3": RATINGS + TIME + M3,
    "win_move10": RATINGS + TIME + M10,
    "win_move10_engine": RATINGS + TIME + M10 + ["m10_engine"],  # Stockfish on the move-10 board
    "elo_move10": TIME + M10,                      # predicts elo_w and elo_b: no rating may enter
    "elo_time_only": TIME,                         # the same target, to see what each part adds
    "elo_moves_only": M10,
}
TARGET = {"win_pregame": "score", "win_move3": "score", "win_move10": "score",
          "win_move10_engine": "score", "elo_move10": ["elo_w", "elo_b"],
          "elo_time_only": ["elo_w", "elo_b"], "elo_moves_only": ["elo_w", "elo_b"]}
NEEDS = {"win_pregame": None, "win_move3": "m3_material_diff", "win_move10": "m10_material_diff",
         "win_move10_engine": "m10_material_diff",
         "elo_move10": "m10_material_diff", "elo_time_only": "m10_material_diff",
         "elo_moves_only": "m10_material_diff"}    # a game is eligible when this is not NaN
CATEGORICAL = ["first_move"]


def eligible(df, model):
    """Games a model is trained and scored on: played games that reach its horizon.

    Played means each side made at least one move; this drops every abandoned
    game and the games that ended with no move at all."""
    keep = df.plies >= 2
    if NEEDS[model]:
        keep &= df[NEEDS[model]].notna()
    return df[keep]
