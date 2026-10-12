"""What a leak looks like: give the Elo model one rating-derived column.

elo_diff looks harmless (it is not a rating), but with it the model only has
to learn one rating to know both. The score improves, and the contract test
catches it.
"""

import numpy as np

import contract
import test_contract
from contract import eligible
from models import fit_predict, load, split

f = load()
train, valid = split(f)
truth = eligible(valid, "elo_move10")[["elo_w", "elo_b"]].to_numpy()
for label, cols in (("as contracted", contract.FEATURES["elo_move10"]),
                    ("plus elo_diff", contract.FEATURES["elo_move10"] + ["elo_diff"])):
    _, p = fit_predict("elo_move10", train, valid, cols=cols)
    print(f"Elo model {label:14s} MAE {np.abs(truth - p).mean():6.1f} rating points")

contract.FEATURES["elo_move10"] = contract.FEATURES["elo_move10"] + ["elo_diff"]
try:
    test_contract.test_elo_model_sees_no_rating()
except AssertionError:
    print("test_elo_model_sees_no_rating: FAILED, the leak is caught before any score is trusted")
