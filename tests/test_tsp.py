import itertools

import numpy as np
import pytest

from lastmile import tsp


def random_costs(n: int, seed: int) -> np.ndarray:
    """Road-like costs: distances between random points, made slightly asymmetric."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 100, size=(n, 2))
    cost = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    cost = cost * rng.uniform(0.9, 1.1, size=(n, n))
    np.fill_diagonal(cost, 0.0)
    return cost


def brute_force(cost: np.ndarray) -> float:
    n = len(cost)
    return min(
        tsp.tour_cost(cost, np.array((0,) + order))
        for order in itertools.permutations(range(1, n))
    )


@pytest.mark.parametrize("seed", range(8))
def test_finds_optimal_tour_on_small_instances(seed):
    cost = random_costs(8, seed)
    tour = tsp.solve(cost, time_limit=0.2, seed=seed)
    assert tour[0] == 0
    assert sorted(tour.tolist()) == list(range(8))
    assert tsp.tour_cost(cost, tour) == pytest.approx(brute_force(cost))


@pytest.mark.parametrize("seed", range(5))
def test_moves_report_their_true_cost_change(seed):
    """Each move's predicted change must equal the recomputed tour cost."""
    cost = random_costs(30, seed)
    rng = np.random.default_rng(seed)
    tour = np.concatenate(([0], rng.permutation(np.arange(1, 30))))
    before = tsp.tour_cost(cost, tour)

    delta, i, j = tsp._best_two_opt(cost, tour)
    reversed_tour = tour.copy()
    reversed_tour[i : j + 1] = reversed_tour[i : j + 1][::-1]
    assert tsp.tour_cost(cost, reversed_tour) - before == pytest.approx(delta)

    for length in (1, 2, 3):
        delta, start, k = tsp._best_segment_move(cost, tour, length)
        moved = tsp._apply_segment_move(tour, start, length, k)
        assert sorted(moved.tolist()) == list(range(30))
        assert moved[0] == 0
        assert tsp.tour_cost(cost, moved) - before == pytest.approx(delta)


def test_local_search_never_makes_a_tour_worse():
    cost = random_costs(60, 1)
    start = tsp.nearest_neighbour_tour(cost)
    improved = tsp.local_search(cost, start)
    assert tsp.tour_cost(cost, improved) <= tsp.tour_cost(cost, start)
