"""Simple sequencing methods that use travel times only.

They ignore how drivers behave, so they mark the level a learned model has to beat.
Every method returns stop numbers starting with 0, the station.
"""

from __future__ import annotations

import random

from . import tsp
from .data import Route


def random_order(route: Route, seed: int = 0) -> list[int]:
    """Stops in random order: the floor any method should clear."""
    stops = list(range(1, route.n_stops))
    random.Random(seed).shuffle(stops)
    return [0] + stops


def nearest_neighbour(route: Route) -> list[int]:
    """From the station, always drive to the closest unvisited stop."""
    return tsp.nearest_neighbour_tour(route.times).tolist()


def shortest_route(route: Route, rounds: int = 20, seed: int = 0) -> list[int]:
    """Shortest closed route on travel time alone."""
    return tsp.solve(route.times, rounds=rounds, seed=seed).tolist()


def tour_time(route: Route, sequence: list[int]) -> float:
    """Total driving time in seconds, including the return to the station."""
    return tsp.tour_cost(route.times, sequence)
