"""Route scoring for the Amazon Last Mile Routing Research Challenge.

A port of the official scorer (MIT-CAVE/rc-cli, ``scoring/score.py``, kept in
``third_party/rc_cli`` for reference). It returns the same numbers but works on
already-loaded data and replaces the recursive edit-distance with an iterative
one, so full-size routes score quickly. ``tests/test_scoring.py`` checks it
against the official code.

Score of one route (lower is better, 0 is a perfect match)::

    score = sequence_deviation * (ERP distance / number of ERP edits)

A sequence is a list of stop ids that starts at the station and ends with the
station repeated, exactly as the official scorer builds it.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

GAP_PENALTY = 1000

CostMatrix = Mapping[str, Mapping[str, float]]


def sequence_to_list(stops: Mapping[str, int]) -> list[str]:
    """Turn ``{stop_id: position}`` into an ordered list closed at the station."""
    n = len(stops)
    route: list[str | None] = [None] * (n + 1)
    for stop, position in stops.items():
        if type(position) is not int or not 0 <= position < n or route[position] is not None:
            raise ValueError(f"invalid position {position!r} for stop {stop!r}")
        route[position] = stop
    route[-1] = route[0]
    return route  # type: ignore[return-value]


def is_invalid(actual: Sequence[str], proposed: Sequence[str]) -> bool:
    """A proposal must visit the same stops and start from the same station."""
    return (
        len(actual) != len(proposed)
        or set(actual) != set(proposed)
        or actual[0] != proposed[0]
    )


def sequence_deviation(actual: Sequence[str], proposed: Sequence[str]) -> float:
    """How far the proposed order jumps around relative to the actual order."""
    actual = list(actual[1:-1])
    proposed = proposed[1:-1]
    n = len(actual)
    if n < 2:  # the official code divides by zero here; such routes do not occur
        return 0.0
    position = {stop: i for i, stop in enumerate(actual)}
    ranks = [position[stop] for stop in proposed]
    total = sum(abs(ranks[i] - ranks[i - 1]) - 1 for i in range(1, len(ranks)))
    return (2 / (n * (n - 1))) * total


def normalized_times(cost_matrix: CostMatrix) -> dict[str, dict[str, float]]:
    """Standardise travel times, then shift them so the smallest is zero."""
    values = np.array(
        [time for row in cost_matrix.values() for time in row.values()], dtype=float
    )
    shifted = (values - np.mean(values)) / np.std(values)
    shifted = (shifted - shifted.min()).tolist()
    result: dict[str, dict[str, float]] = {}
    k = 0
    for origin, row in cost_matrix.items():
        result[origin] = {}
        for destination in row:
            result[origin][destination] = shifted[k]
            k += 1
    return result


def erp_per_edit(
    actual: Sequence[str],
    proposed: Sequence[str],
    normalized: CostMatrix,
    gap: float = GAP_PENALTY,
) -> float:
    """Edit distance with real penalty, divided by the number of edits made."""
    n, m = len(actual), len(proposed)
    # dist[j] / edits[j] hold the answer for actual[i:] against proposed[j:].
    dist = [(m - j) * gap for j in range(m + 1)]
    edits = [m - j for j in range(m + 1)]
    for i in range(n - 1, -1, -1):
        row = normalized[actual[i]]
        stop = actual[i]
        new_dist = [0.0] * (m + 1)
        new_edits = [0] * (m + 1)
        new_dist[m] = (n - i) * gap
        new_edits[m] = n - i
        for j in range(m - 1, -1, -1):
            other = proposed[j]
            substitute = dist[j + 1] + row[other]
            skip_actual = dist[j] + gap
            skip_proposed = new_dist[j + 1] + gap
            best = min(substitute, skip_actual, skip_proposed)
            # Tie-breaking order matches the official scorer.
            if best == substitute:
                count = edits[j + 1] + (0 if stop == other else 1)
            elif best == skip_actual:
                count = edits[j] + 1
            else:
                count = new_edits[j + 1] + 1
            new_dist[j] = best
            new_edits[j] = count
        dist, edits = new_dist, new_edits
    return dist[0] / edits[0] if edits[0] else 0.0


def score_route(
    actual: Sequence[str],
    proposed: Sequence[str],
    cost_matrix: CostMatrix,
    gap: float = GAP_PENALTY,
) -> float:
    """Score one valid proposed sequence against the driver's actual sequence."""
    deviation = sequence_deviation(actual, proposed)
    return deviation * erp_per_edit(actual, proposed, normalized_times(cost_matrix), gap)


def normalized_matrix(times: np.ndarray) -> np.ndarray:
    """Matrix form of :func:`normalized_times`."""
    shifted = (times - times.mean()) / times.std()
    return shifted - shifted.min()


def score_sequence(actual: Sequence[int], proposed: Sequence[int], times: np.ndarray) -> float:
    """Score a route given as stop numbers into the travel-time matrix ``times``.

    Both sequences start at the station and list every stop once; the return to
    the station is added here.
    """
    actual = list(actual) + [actual[0]]
    proposed = list(proposed) + [proposed[0]]
    if is_invalid(actual, proposed):
        raise ValueError("proposed sequence must visit the same stops from the same station")
    normalized = normalized_matrix(np.asarray(times, dtype=float)).tolist()
    return sequence_deviation(actual, proposed) * erp_per_edit(actual, proposed, normalized)


def evaluate(
    actual_routes: Mapping[str, Mapping[str, Mapping[str, int]]],
    proposed_routes: Mapping[str, Mapping[str, Mapping[str, int]]],
    cost_matrices: Mapping[str, CostMatrix],
    invalid_scores: Mapping[str, float],
) -> dict:
    """Score a whole submission, in the official file layouts.

    ``actual_routes[route] == {"actual": {stop: position}}`` and
    ``proposed_routes[route] == {"proposed": {stop: position}}``. Missing or
    invalid proposals receive that route's pre-computed invalid score.
    """
    route_scores: dict[str, float] = {}
    feasible: dict[str, bool] = {}
    for route, actual_entry in actual_routes.items():
        actual = sequence_to_list(actual_entry["actual"])
        try:
            proposed = sequence_to_list(proposed_routes[route]["proposed"])
        except (KeyError, ValueError, TypeError):
            proposed = None
        if proposed is None or is_invalid(actual, proposed):
            route_scores[route] = invalid_scores[route]
            feasible[route] = False
        else:
            route_scores[route] = score_route(actual, proposed, cost_matrices[route])
            feasible[route] = True
    return {
        "submission_score": float(np.mean(list(route_scores.values()))),
        "route_scores": route_scores,
        "route_feasibility": feasible,
    }
