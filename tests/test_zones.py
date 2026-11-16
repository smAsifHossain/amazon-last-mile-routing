import numpy as np
from helpers import line_route

from lastmile import zones


def test_parse_zone():
    assert zones.parse_zone("D-8.3E") == ("D-8", "3", "E")
    assert zones.parse_zone("B-12.1A") == ("B-12", "1", "A")
    assert zones.parse_zone("unknown") is None


def test_missing_zone_is_taken_from_nearest_stop():
    route = line_route([10.0, 12.0, 50.0], ["A-1.1A", None, "A-1.2A"])
    assert zones.stop_zones(route) == [None, "A-1.1A", "A-1.1A", "A-1.2A"]


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
    stop_zones = ["A-1.1A", "A-1.2A", "A-1.1A", "A-1.2A", "A-1.1A", "A-1.2A"]
    route = line_route([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], stop_zones)
    sequence = zones.zone_aware_route(route)
    assert sequence[0] == 0 and sorted(sequence) == list(range(7))
    visited = [route.zone_ids[stop] for stop in sequence[1:]]
    assert sum(a != b for a, b in zip(visited, visited[1:])) == 1
