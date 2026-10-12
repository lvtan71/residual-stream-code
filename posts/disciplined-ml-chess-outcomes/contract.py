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
    "elo_move10": TIME + M10,                      # predicts elo_w and elo_b: no rating may enter
}
TARGET = {"win_pregame": "score", "win_move3": "score", "win_move10": "score",
          "elo_move10": ["elo_w", "elo_b"]}
NEEDS = {"win_pregame": None, "win_move3": "m3_material_diff", "win_move10": "m10_material_diff",
         "elo_move10": "m10_material_diff"}        # a game is eligible when this is not NaN
CATEGORICAL = ["first_move"]


def eligible(df, model):
    """Games a model is trained and scored on: played games that reach its horizon."""
    keep = df.termination != "Abandoned"
    if NEEDS[model]:
        keep &= df[NEEDS[model]].notna()
    return df[keep]
