"""Travelling-salesman local search for asymmetric cost matrices.

Finds a short closed tour that starts and ends at node 0. It is a heuristic
(iterated local search with 2-opt and segment moves), not an exact solver: tours
are typically within a few percent of optimal for routes of 100-200 stops.
"""

from __future__ import annotations

import time

import numpy as np

EPS = 1e-9


def tour_cost(cost: np.ndarray, tour: np.ndarray) -> float:
    """Cost of the closed tour ``tour[0] -> ... -> tour[-1] -> tour[0]``."""
    return float(cost[tour, np.roll(tour, -1)].sum())


def nearest_neighbour_tour(cost: np.ndarray) -> np.ndarray:
    n = len(cost)
    unvisited = np.ones(n, dtype=bool)
    unvisited[0] = False
    tour = [0]
    for _ in range(n - 1):
        row = np.where(unvisited, cost[tour[-1]], np.inf)
        nxt = int(row.argmin())
        unvisited[nxt] = False
        tour.append(nxt)
    return np.array(tour)


def _best_two_opt(cost: np.ndarray, tour: np.ndarray):
    """Best reversal of ``tour[i..j]`` (1 <= i < j); exact for asymmetric costs."""
    n = len(tour)
    nxt = np.roll(tour, -1)
    forward = cost[tour, nxt]  # edge p -> p+1
    backward = cost[nxt, tour]  # the same edge driven the other way
    cf = np.concatenate(([0.0], np.cumsum(forward)))
    cb = np.concatenate(([0.0], np.cumsum(backward)))
    pos = np.arange(1, n)
    before = tour[pos - 1]  # node before the segment, per i
    node = tour[pos]
    after = nxt[pos]  # node after the segment, per j
    delta = (
        cost[before[:, None], node[None, :]]
        + cost[node[:, None], after[None, :]]
        - cost[before, node][:, None]
        - cost[node, after][None, :]
        + (cb[pos][None, :] - cb[pos][:, None])
        - (cf[pos][None, :] - cf[pos][:, None])
    )
    delta[np.tril_indices(n - 1)] = np.inf  # keep i < j only
    flat = int(delta.argmin())
    i, j = divmod(flat, n - 1)
    return float(delta[i, j]), i + 1, j + 1


def _best_segment_move(cost: np.ndarray, tour: np.ndarray, length: int):
    """Best move of ``tour[i..i+length-1]`` to between ``tour[k]`` and ``tour[k+1]``."""
    n = len(tour)
    if n - 1 <= length:
        return np.inf, 0, 0
    nxt = np.roll(tour, -1)
    starts = np.arange(1, n - length + 1)
    before = tour[starts - 1]
    first = tour[starts]
    last = tour[starts + length - 1]
    after = nxt[starts + length - 1]
    removal = cost[before, first] + cost[last, after] - cost[before, after]
    k = np.arange(n)
    insertion = (
        cost[tour[None, :], first[:, None]]
        + cost[last[:, None], nxt[None, :]]
        - cost[tour, nxt][None, :]
    )
    delta = insertion - removal[:, None]
    # Positions inside or adjacent to the segment are not real moves.
    blocked = (k[None, :] >= starts[:, None] - 1) & (k[None, :] <= starts[:, None] + length - 1)
    delta[blocked] = np.inf
    flat = int(delta.argmin())
    row, col = divmod(flat, n)
    return float(delta[row, col]), int(starts[row]), col


def _apply_segment_move(tour: np.ndarray, start: int, length: int, k: int) -> np.ndarray:
    segment = tour[start : start + length]
    rest = np.concatenate((tour[:start], tour[start + length :]))
    insert_at = k + 1 if k < start else k - length + 1
    return np.concatenate((rest[:insert_at], segment, rest[insert_at:]))


def local_search(cost: np.ndarray, tour: np.ndarray) -> np.ndarray:
    """Apply the best improving move until none is left."""
    tour = tour.copy()
    while True:
        best_delta, i, j = _best_two_opt(cost, tour)
        move = ("reverse", i, j)
        for length in (1, 2, 3):
            delta, start, k = _best_segment_move(cost, tour, length)
            if delta < best_delta:
                best_delta, move = delta, ("move", start, length, k)
        if best_delta >= -EPS:
            return tour
        if move[0] == "reverse":
            _, i, j = move
            tour[i : j + 1] = tour[i : j + 1][::-1]
        else:
            _, start, length, k = move
            tour = _apply_segment_move(tour, start, length, k)


def _perturb(tour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Double bridge: swap two neighbouring segments, keeping their direction."""
    n = len(tour)
    a, b, c = np.sort(rng.choice(np.arange(1, n), size=3, replace=False))
    return np.concatenate((tour[:a], tour[b:c], tour[a:b], tour[c:]))


def solve(cost: np.ndarray, time_limit: float = 1.0, seed: int = 0) -> np.ndarray:
    """Return node indices of a short closed tour starting at node 0."""
    cost = np.asarray(cost, dtype=float)
    n = len(cost)
    if n <= 3:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    best = local_search(cost, nearest_neighbour_tour(cost))
    best_cost = tour_cost(cost, best)
    deadline = time.perf_counter() + time_limit
    while time.perf_counter() < deadline:
        candidate = local_search(cost, _perturb(best, rng))
        candidate_cost = tour_cost(cost, candidate)
        if candidate_cost < best_cost - EPS:
            best, best_cost = candidate, candidate_cost
    return best
