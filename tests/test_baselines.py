from helpers import line_route

from lastmile import baselines


def test_nearest_neighbour_walks_the_line():
    route = line_route([1.0, 2.0, 3.0])
    assert baselines.nearest_neighbour(route) == [0, 1, 2, 3]
    assert baselines.tour_time(route, [0, 1, 2, 3]) == 6.0


def test_random_order_starts_at_station_and_visits_everything():
    sequence = baselines.random_order(line_route([1.0, 2.0, 3.0]), seed=1)
    assert sequence[0] == 0
    assert sorted(sequence) == [0, 1, 2, 3]


def test_shortest_route_is_reproducible():
    route = line_route([5.0, 1.0, 9.0, 3.0, 7.0, 2.0])
    first = baselines.shortest_route(route)
    assert first == baselines.shortest_route(route)
    assert baselines.tour_time(route, first) == 18.0
