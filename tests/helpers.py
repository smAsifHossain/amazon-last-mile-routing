"""Small hand-made routes for tests."""

from __future__ import annotations

import numpy as np

from lastmile.data import Route


def line_route(positions: list[float], zones: list[str | None] | None = None, actual=None) -> Route:
    """Station at 0 and drop-offs at ``positions`` on a line; travel time = distance."""
    x = np.array([0.0] + list(positions))
    n = len(x)
    return Route(
        id="RouteID_test",
        station_code="TEST",
        date="2018-01-01",
        departure_time_utc="08:00:00",
        capacity_cm3=1.0,
        quality=None,
        stop_ids=["SS"] + [chr(65 + i) * 2 for i in range(n - 1)],
        lat=np.zeros(n),
        lng=x,
        zone_ids=[None] + (list(zones) if zones else [None] * (n - 1)),
        times=np.abs(x[:, None] - x[None, :]),
        n_packages=np.ones(n, dtype=np.int32),
        service_time=np.zeros(n),
        window_start=np.full(n, np.nan),
        window_end=np.full(n, np.nan),
        actual=actual,
    )
