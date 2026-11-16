"""Check our scorer against the official challenge scorer."""

import copy
import importlib.util
import random
import string
from pathlib import Path

import pytest

from lastmile import scoring

OFFICIAL_PATH = Path(__file__).resolve().parents[1] / "third_party" / "rc_cli" / "score.py"


def load_official():
    spec = importlib.util.spec_from_file_location("official_score", OFFICIAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


official = load_official()


def random_route(rng: random.Random, n_stops: int):
    """A station plus ``n_stops`` stops with an asymmetric travel-time matrix."""
    ids = rng.sample([a + b for a in string.ascii_uppercase for b in string.ascii_uppercase], n_stops + 1)
    station, stops = ids[0], ids[1:]
    cost = {
        a: {b: 0.0 if a == b else round(rng.uniform(5, 900), 1) for b in ids} for a in ids
    }
    actual_order = [station] + rng.sample(stops, n_stops)
    proposed_order = [station] + rng.sample(stops, n_stops)
    actual = {"actual": {stop: i for i, stop in enumerate(actual_order)}}
    proposed = {"proposed": {stop: i for i, stop in enumerate(proposed_order)}}
    return actual, proposed, cost


@pytest.mark.parametrize("seed", range(60))
def test_matches_official_scorer(seed):
    rng = random.Random(seed)
    actual, proposed, cost = random_route(rng, rng.randint(3, 14))
    actual_list = scoring.sequence_to_list(actual["actual"])
    proposed_list = scoring.sequence_to_list(proposed["proposed"])
    assert actual_list == official.route2list(actual)
    assert proposed_list == official.route2list(proposed)

    # The official scorer overwrites the matrix it is given, so hand it a copy.
    expected = official.score(actual_list, proposed_list, copy.deepcopy(cost))
    assert scoring.score_route(actual_list, proposed_list, cost) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_near_identical_sequences_match_official(seed):
    """Small perturbations exercise the tie-breaking in the edit distance."""
    rng = random.Random(1000 + seed)
    actual, _, cost = random_route(rng, 12)
    order = scoring.sequence_to_list(actual["actual"])[:-1]
    i = rng.randint(1, len(order) - 2)
    swapped = order[:]
    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
    actual_list = order + [order[0]]
    proposed_list = swapped + [swapped[0]]
    expected = official.score(actual_list, proposed_list, copy.deepcopy(cost))
    assert scoring.score_route(actual_list, proposed_list, cost) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_perfect_match_scores_zero():
    actual, _, cost = random_route(random.Random(7), 10)
    order = scoring.sequence_to_list(actual["actual"])
    assert scoring.score_route(order, order, cost) == 0.0


def test_evaluate_uses_invalid_score_for_bad_proposals():
    rng = random.Random(3)
    a1, p1, c1 = random_route(rng, 6)
    a2, p2, c2 = random_route(rng, 6)
    a3, p3, c3 = random_route(rng, 6)
    # Route 2: wrong starting stop. Route 3: missing from the submission.
    stops = list(p2["proposed"])
    p2["proposed"][stops[0]], p2["proposed"][stops[1]] = p2["proposed"][stops[1]], p2["proposed"][stops[0]]
    actual = {"RouteID_1": a1, "RouteID_2": a2, "RouteID_3": a3}
    proposed = {"RouteID_1": p1, "RouteID_2": p2}
    costs = {"RouteID_1": c1, "RouteID_2": c2, "RouteID_3": c3}
    invalid = {"RouteID_1": 0.9, "RouteID_2": 0.8, "RouteID_3": 0.7}

    result = scoring.evaluate(actual, proposed, costs, invalid)

    assert result["route_feasibility"] == {"RouteID_1": True, "RouteID_2": False, "RouteID_3": False}
    assert result["route_scores"]["RouteID_2"] == 0.8
    assert result["route_scores"]["RouteID_3"] == 0.7
    expected = (result["route_scores"]["RouteID_1"] + 0.8 + 0.7) / 3
    assert result["submission_score"] == pytest.approx(expected)
