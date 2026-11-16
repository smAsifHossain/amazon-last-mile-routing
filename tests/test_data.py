import json

import numpy as np
import pytest

from lastmile import data, scoring

RAW_ROUTES = {
    "RouteID_a": {
        "station_code": "DXX1",
        "date_YYYY_MM_DD": "2018-08-11",
        "departure_time_utc": "15:00:00",
        "executor_capacity_cm3": 100.0,
        "route_score": "High",
        "stops": {
            "AA": {"lat": 1.0, "lng": 2.0, "type": "Dropoff", "zone_id": "A-1.1A"},
            "ST": {"lat": 0.0, "lng": 0.0, "type": "Station", "zone_id": float("nan")},
            "BB": {"lat": 3.0, "lng": 4.0, "type": "Dropoff", "zone_id": float("nan")},
        },
    },
    "RouteID_b": {
        "station_code": "DXX1",
        "date_YYYY_MM_DD": "2018-08-12",
        "departure_time_utc": "16:00:00",
        "executor_capacity_cm3": 200.0,
        "stops": {
            "ST": {"lat": 0.0, "lng": 0.0, "type": "Station", "zone_id": float("nan")},
            "AA": {"lat": 1.0, "lng": 2.0, "type": "Dropoff", "zone_id": "B-2.1C"},
        },
    },
}
RAW_TIMES = {
    "RouteID_a": {
        "AA": {"AA": 0.0, "ST": 10.5, "BB": 20.1},
        "ST": {"AA": 11.5, "ST": 0.0, "BB": 30.2},
        "BB": {"AA": 21.1, "ST": 31.2, "BB": 0.0},
    },
    "RouteID_b": {"ST": {"ST": 0.0, "AA": 5.0}, "AA": {"ST": 6.0, "AA": 0.0}},
}
RAW_PACKAGES = {
    "RouteID_a": {
        "AA": {
            "P1": {
                "time_window": {"start_time_utc": "2018-08-11 16:00:00", "end_time_utc": "2018-08-11 20:00:00"},
                "planned_service_time_seconds": 30.0,
            },
            "P2": {
                "time_window": {"start_time_utc": "2018-08-11 17:00:00", "end_time_utc": "2018-08-11 21:00:00"},
                "planned_service_time_seconds": 45.0,
            },
        },
        "ST": {},
        "BB": {"P3": {"time_window": {"start_time_utc": float("nan"), "end_time_utc": float("nan")}, "planned_service_time_seconds": 60.0}},
    },
    "RouteID_b": {"ST": {}, "AA": {}},
}
RAW_ACTUAL = {"RouteID_a": {"actual": {"AA": 2, "ST": 0, "BB": 1}}}


@pytest.fixture
def raw_files(tmp_path):
    files = {}
    for name, content in [("routes", RAW_ROUTES), ("times", RAW_TIMES), ("packages", RAW_PACKAGES), ("actual", RAW_ACTUAL)]:
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(content), encoding="utf-8")
    return files


def test_streaming_reader_matches_json_load(tmp_path):
    content = {f"key{i}": {"values": list(range(i)), "text": "x" * (i * 7)} for i in range(200)}
    path = tmp_path / "big.json"
    path.write_text(json.dumps(content, indent=1), encoding="utf-8")
    # A tiny chunk size forces entries to straddle many buffer refills.
    assert dict(data.iter_json_object(path, chunk_size=64)) == content


def test_route_from_raw_files(raw_files):
    a, b = data.iter_raw_routes(raw_files["routes"], raw_files["times"], raw_files["packages"], raw_files["actual"])
    assert a.stop_ids == ["ST", "AA", "BB"]  # station first
    assert a.zone_ids == [None, "A-1.1A", None]
    assert a.quality == "High"
    assert a.times.tolist() == [[0.0, 11.5, 30.2], [10.5, 0.0, 20.1], [31.2, 21.1, 0.0]]
    assert a.n_packages.tolist() == [0, 2, 1]
    assert a.service_time.tolist() == [0.0, 75.0, 60.0]
    assert a.window_start[1] == 2 * 3600 and a.window_end[1] == 5 * 3600  # tightest window
    assert np.isnan(a.window_start[2])
    assert a.actual == [0, 2, 1]
    assert b.actual is None and b.quality is None


def test_save_and_load_round_trip(raw_files, tmp_path):
    original = next(data.iter_raw_routes(raw_files["routes"], raw_files["times"], raw_files["packages"], raw_files["actual"]))
    loaded = data.load_route(data.save_route(original, tmp_path / "out"))
    assert loaded.id == original.id and loaded.quality == original.quality
    assert loaded.stop_ids == original.stop_ids and loaded.zone_ids == original.zone_ids
    assert np.array_equal(loaded.times, original.times)  # exact, not approximate
    assert np.array_equal(loaded.window_end, original.window_end, equal_nan=True)
    assert loaded.actual == original.actual


def test_matrix_scoring_matches_official_layout_scoring():
    rng = np.random.default_rng(0)
    ids = ["ST", "AA", "BB", "CC", "DD", "EE", "FF"]
    raw = {
        "station_code": "DXX1",
        "date_YYYY_MM_DD": "2018-08-11",
        "departure_time_utc": "15:00:00",
        "executor_capacity_cm3": 100.0,
        "stops": {s: {"lat": 0.0, "lng": 0.0, "type": "Station" if s == "ST" else "Dropoff", "zone_id": "A-1.1A"} for s in ids},
    }
    times = {a: {b: 0.0 if a == b else round(float(rng.uniform(10, 600)), 1) for b in ids} for a in ids}
    actual = {"actual": {"ST": 0, "CC": 1, "AA": 2, "FF": 3, "BB": 4, "EE": 5, "DD": 6}}
    route = data.route_from_raw("RouteID_x", raw, times, None, actual)
    proposed = [0, 1, 2, 3, 4, 5, 6]

    by_matrix = scoring.score_sequence(route.actual, proposed, route.times)
    submission = data.to_submission([route], [proposed])
    by_files = scoring.evaluate({"RouteID_x": actual}, submission, {"RouteID_x": times}, {"RouteID_x": 1.0})

    assert by_files["route_feasibility"]["RouteID_x"]
    assert by_matrix > 0
    assert by_matrix == pytest.approx(by_files["route_scores"]["RouteID_x"], rel=1e-9)
