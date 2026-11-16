"""Simple sequencing methods that use travel times only.

They ignore how drivers behave, so they mark the level a learned model has to beat.
"""

from __future__ import annotations

import random

from .data import Route


def random_order(route: Route, seed: int = 0) -> list[str]:
    """Stops in random order: the floor any method should clear."""
    stops = route.dropoffs
    random.Random(seed).shuffle(stops)
    return [route.station] + stops


def nearest_neighbour(route: Route) -> list[str]:
    """From the station, always drive to the closest unvisited stop."""
    times = route.travel_times
    current = route.station
    remaining = set(route.dropoffs)
    sequence = [current]
    while remaining:
        row = times[current]
        current = min(remaining, key=lambda stop: (row[stop], stop))
        remaining.remove(current)
        sequence.append(current)
    return sequence


def tour_time(route: Route, sequence: list[str]) -> float:
    """Total driving time in seconds, including the return to the station."""
    times = route.travel_times
    closed = sequence + [sequence[0]]
    return sum(times[a][b] for a, b in zip(closed, closed[1:]))
