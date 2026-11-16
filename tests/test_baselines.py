from lastmile import baselines
from lastmile.data import Route, Stop, to_submission


def tiny_route() -> Route:
    # Station S and three stops on a line: S - A - B - C.
    position = {"SS": 0, "AA": 1, "BB": 2, "CC": 3}
    stops = {
        name: Stop(id=name, lat=0.0, lng=float(x), is_station=name == "SS", zone_id=None)
        for name, x in position.items()
    }
    times = {a: {b: float(abs(position[a] - position[b])) for b in position} for a in position}
    return Route("RouteID_test", "TEST", "2018-01-01", "00:00:00", 1.0, None, stops, times)


def test_nearest_neighbour_walks_the_line():
    route = tiny_route()
    assert baselines.nearest_neighbour(route) == ["SS", "AA", "BB", "CC"]
    assert baselines.tour_time(route, ["SS", "AA", "BB", "CC"]) == 6.0


def test_random_order_starts_at_station_and_visits_everything():
    route = tiny_route()
    sequence = baselines.random_order(route, seed=1)
    assert sequence[0] == "SS"
    assert sorted(sequence[1:]) == ["AA", "BB", "CC"]


def test_to_submission_layout():
    assert to_submission({"RouteID_test": ["SS", "BB", "AA"]}) == {
        "RouteID_test": {"proposed": {"SS": 0, "BB": 1, "AA": 2}}
    }
