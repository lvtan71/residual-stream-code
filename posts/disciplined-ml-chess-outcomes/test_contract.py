"""Tests that turn the leakage contract into something a refactor cannot break.

Run with `python test_contract.py` (or pytest). Each test is one rule.
"""

from contract import FEATURES, M3, M10, RATINGS, TARGET


def test_pregame_sees_no_moves():
    assert not any(c.startswith(("m3_", "m10_")) or c == "first_move" for c in FEATURES["win_pregame"])


def test_move3_sees_nothing_after_move3():
    assert not any(c.startswith("m10_") for c in FEATURES["win_move3"])


def test_horizons_only_add_information():
    assert set(FEATURES["win_pregame"]) < set(FEATURES["win_move3"]) < set(FEATURES["win_move10"])


def test_engine_sees_only_the_move10_board():
    assert set(FEATURES["win_move10_engine"]) - set(FEATURES["win_move10"]) == {"m10_engine"}


def test_elo_model_sees_no_rating():
    for model, cols in FEATURES.items():
        if TARGET[model] == ["elo_w", "elo_b"]:
            assert not set(cols) & set(RATINGS), model


def test_no_model_sees_its_target():
    for model, cols in FEATURES.items():
        target = TARGET[model]
        assert not set(cols) & set([target] if isinstance(target, str) else target), model


def test_no_outcome_or_identity_columns():
    for cols in FEATURES.values():                 # results, how the game ended, ids, names
        assert not {"score", "termination", "plies", "game", "white", "black"} & set(cols)


def test_move_groups_are_disjoint_from_ratings():
    assert not set(M3 + M10) & set(RATINGS)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} rules hold")
