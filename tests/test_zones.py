import numpy as np

from lastmile import zones
from lastmile.data import Route, Stop


def make_route(stop_zones: dict[str, str | None], positions: dict[str, float]) -> Route:
    stops = {"SS": Stop("SS", 0.0, 0.0, True, None)}
    for name, zone in stop_zones.items():
        stops[name] = Stop(name, 0.0, positions[name], False, zone)
    x = {"SS": 0.0, **positions}
    times = {a: {b: abs(x[a] - x[b]) for b in x} for a in x}
    return Route("RouteID_test", "TEST", "2018-01-01", "00:00:00", 1.0, None, stops, times)


def test_parse_zone():
    assert zones.parse_zone("D-8.3E") == ("D-8", "3", "E")
    assert zones.parse_zone("B-12.1A") == ("B-12", "1", "A")
    assert zones.parse_zone("unknown") is None


def test_missing_zone_is_taken_from_nearest_stop():
    route = make_route(
        {"AA": "A-1.1A", "BB": None, "CC": "A-1.2A"},
        {"AA": 10.0, "BB": 12.0, "CC": 50.0},
    )
    assert zones.stop_zones(route, ["SS", "AA", "BB", "CC"]) == [None, "A-1.1A", "A-1.1A", "A-1.2A"]


def test_penalties_grow_with_how_much_the_zone_changes():
    p = zones.ZonePenalties(different_zone=1, both_parts=10, different_region=100)
    ids = [None, "A-1.1A", "A-1.1A", "A-1.2A", "A-1.2B", "A-2.1A"]
    m = zones.penalty_matrix(ids, p)
    assert np.all(m[0] == 0) and np.all(m[:, 0] == 0)  # station
    assert m[1, 2] == 0  # same zone
    assert m[1, 3] == 1  # only the sub-area changes
    assert m[3, 4] == 1  # only the block changes
    assert m[1, 4] == 11  # both change
    assert m[1, 5] == 101  # region changes
    assert np.array_equal(m, m.T)


def test_zone_aware_route_keeps_zones_together():
    # Along a line the zones alternate, so the shortest route mixes them.
    route = make_route(
        {"AA": "A-1.1A", "BB": "A-1.2A", "CC": "A-1.1A", "DD": "A-1.2A", "EE": "A-1.1A", "FF": "A-1.2A"},
        {"AA": 10.0, "BB": 11.0, "CC": 12.0, "DD": 13.0, "EE": 14.0, "FF": 15.0},
    )
    sequence = zones.zone_aware_route(route, time_limit=0.2)
    assert sequence[0] == "SS" and sorted(sequence[1:]) == ["AA", "BB", "CC", "DD", "EE", "FF"]
    visited = [route.stops[stop].zone_id for stop in sequence[1:]]
    changes = sum(a != b for a, b in zip(visited, visited[1:]))
    assert changes == 1
