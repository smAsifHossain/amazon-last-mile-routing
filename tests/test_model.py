import numpy as np
from helpers import line_route

from lastmile import baselines, model, tsp

ZONES = ["A-1.1A", "A-1.2A", "A-1.3A", "A-1.4A"]


def far_first_route(seed: int, with_actual: bool = True):
    """Four zones along a line; the driver always serves the farthest zone first.

    On a line both directions are equally short, so travel time alone cannot
    tell which way the driver goes.
    """
    rng = np.random.default_rng(seed)
    positions, stop_zones = [], []
    for z, centre in enumerate([100.0, 200.0, 300.0, 400.0]):
        for offset in np.sort(rng.uniform(-20, 20, size=3)):
            positions.append(centre + offset)
            stop_zones.append(ZONES[z])
    actual = [0] + sorted(range(1, 13), key=lambda stop: -positions[stop - 1]) if with_actual else None
    return line_route(positions, stop_zones, actual)


def test_fit_choice_learns_which_feature_drives_the_choice():
    rng = np.random.default_rng(0)
    rows, sizes = [], []
    for _ in range(200):
        group = rng.normal(size=(5, 2))
        group = group[np.argsort(-group[:, 0])]  # chosen (first) row has the largest feature 0
        rows.append(group)
        sizes.append(5)
    weights = model.fit_choice(np.concatenate(rows), np.array(sizes))
    assert weights[0] > 1.0
    assert abs(weights[1]) < 0.5 * weights[0]


def test_zone_view_summarises_a_route():
    view = model.zone_view(far_first_route(0))
    assert view.zones == ZONES
    assert view.stop_zone.tolist() == [-1] + [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3
    assert view.order == [3, 2, 1, 0]
    assert view.from_station[0] < view.from_station[3]
    assert view.min_time[0, 1] < view.mean_time[0, 1] < view.mean_time[0, 3]


def test_history_can_leave_out_a_routes_own_moves():
    views = [model.zone_view(far_first_route(seed)) for seed in range(3)]
    history = model.History()
    for view in views:
        history.add(view)
    pair, first, last = history.counts(views[0], exclude_own=False)
    assert pair[3, 2] == 3 and pair[2, 3] == 0 and first[3] == 3 and last[0] == 3
    pair, first, last = history.counts(views[0], exclude_own=True)
    assert pair[3, 2] == 2 and first[3] == 2 and last[0] == 2


def test_sequence_by_zones_follows_the_given_zone_order():
    rng = np.random.default_rng(5)
    for seed in range(5):
        route = far_first_route(seed)
        # Realistic travel times: asymmetric, one decimal, as in the dataset.
        route.times = np.round(route.times * rng.uniform(0.8, 1.2, route.times.shape) + 37.3, 1)
        np.fill_diagonal(route.times, 0.0)
        view = model.zone_view(route)
        for order in ([2, 0, 3, 1], [3, 2, 1, 0], [0, 1, 2, 3]):
            sequence = model.sequence_by_zones(route, view, order)
            assert sequence[0] == 0 and sorted(sequence) == list(range(13))
            visited = [int(view.stop_zone[stop]) for stop in sequence[1:]]
            assert list(dict.fromkeys(visited)) == order
            assert sum(a != b for a, b in zip(visited, visited[1:])) == 3  # each zone served once


def test_local_search_stops_when_the_matrix_holds_very_large_costs():
    # Regression: rounding noise from large costs used to look like endless tiny gains.
    rng = np.random.default_rng(0)
    cost = np.round(rng.uniform(30, 900, size=(60, 60)), 1)
    cost[rng.random((60, 60)) < 0.5] += 1e6
    np.fill_diagonal(cost, 0.0)
    start = np.arange(60)
    result = tsp.local_search(cost, start)
    assert sorted(result.tolist()) == list(range(60))
    assert tsp.tour_cost(cost, result) <= tsp.tour_cost(cost, start)


def test_model_learns_the_direction_drivers_take():
    fitted = model.ZoneModel.fit([model.zone_view(far_first_route(seed)) for seed in range(12)])
    new_route = far_first_route(99, with_actual=False)

    sequence = fitted.predict(new_route)

    assert sequence[0] == 0 and sorted(sequence) == list(range(13))
    visited = list(dict.fromkeys(new_route.zone_ids[stop] for stop in sequence[1:]))
    assert visited == ZONES[::-1]  # farthest zone first, as the drivers did
    # ... at no extra driving: it is as short as the shortest route.
    assert baselines.tour_time(new_route, sequence) == baselines.tour_time(
        new_route, baselines.shortest_route(new_route)
    )
