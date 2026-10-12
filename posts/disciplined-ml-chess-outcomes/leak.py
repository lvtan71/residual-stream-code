"""What a leak looks like: someone adds one rating-derived column to the
rating model's entry in the contract.

elo_diff looks harmless (it is not a rating), but with it the model only has
to learn one rating to know both. The contract test fails on the edit; the
score the leaked model would have reported is shown anyway.
"""

import numpy as np

import contract
import test_contract
from contract import eligible
from models import fit_predict, load, split

contract.FEATURES["elo_move10"] = contract.FEATURES["elo_move10"] + ["elo_diff"]   # the edit
try:
    test_contract.test_elo_model_sees_no_rating()
    print("test_elo_model_sees_no_rating: passed")
except AssertionError:
    print("test_elo_model_sees_no_rating: FAILED on the edited contract")

f = load()
train, valid = split(f)
truth = eligible(valid, "elo_move10")[["elo_w", "elo_b"]].to_numpy()
_, leaked = fit_predict("elo_move10", train, valid)
contract.FEATURES["elo_move10"].remove("elo_diff")                                  # undo it
_, honest = fit_predict("elo_move10", train, valid)
print(f"rating model as contracted  MAE {np.abs(truth - honest).mean():6.1f} rating points")
print(f"rating model with elo_diff  MAE {np.abs(truth - leaked).mean():6.1f} rating points")
