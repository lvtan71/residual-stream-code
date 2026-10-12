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


def test_elo_model_sees_no_rating():
    assert not set(FEATURES["elo_move10"]) & set(RATINGS)


def test_no_model_sees_its_target():
    for model, cols in FEATURES.items():
        target = TARGET[model]
        assert not set(cols) & set([target] if isinstance(target, str) else target), model


def test_no_post_game_columns():
    for cols in FEATURES.values():
        assert not {"score", "termination", "game", "fen10"} & set(cols)


def test_move_groups_are_disjoint_from_ratings():
    assert not set(M3 + M10) & set(RATINGS)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} rules hold")
