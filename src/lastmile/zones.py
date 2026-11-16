"""Delivery zones and the zone-aware sequencing method.

Every drop-off carries a zone id such as ``D-8.3E``: a region (``D-8``), a
sub-area number (``3``) and a block letter (``E``). In the driver sequences,
stops of one zone are almost always served together, and the next zone usually
differs from the current one in only the sub-area number or only the block
letter. The method here encodes that as extra cost on the travel-time matrix and
then solves for the shortest route under the adjusted costs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from . import tsp
from .data import Route

ZONE_PATTERN = re.compile(r"^([A-Z]+)-(\d+)\.(\d+)([A-Z]+)$")


@dataclass(frozen=True)
class ZonePenalties:
    """Seconds added to the travel time between two stops, by how their zones differ."""

    different_zone: float = 600.0  # leaving a zone at all
    both_parts: float = 600.0  # sub-area number and block letter both change
    different_region: float = 1200.0  # region changes


def parse_zone(zone_id: str) -> tuple[str, str, str] | None:
    """Split ``D-8.3E`` into ``("D-8", "3", "E")``; None if the id is unusual."""
    match = ZONE_PATTERN.match(zone_id)
    if not match:
        return None
    letter, number, sub_area, block = match.groups()
    return f"{letter}-{number}", sub_area, block


def stop_zones(route: Route) -> list[str | None]:
    """Zone of each stop; stops without one take the zone of the nearest stop that has one."""
    zoned = [i for i in range(1, route.n_stops) if route.zone_ids[i]]
    zones: list[str | None] = [None]  # the station
    for i in range(1, route.n_stops):
        if route.zone_ids[i] or not zoned:
            zones.append(route.zone_ids[i])
        else:
            nearest = min(zoned, key=lambda other: (route.times[i, other], other))
            zones.append(route.zone_ids[nearest])
    return zones


def penalty_matrix(zones: list[str | None], penalties: ZonePenalties) -> np.ndarray:
    """Extra cost between every pair of stops. Trips to or from the station are free."""
    distinct = sorted({zone for zone in zones if zone})
    index = {zone: i for i, zone in enumerate(distinct)}
    parsed = [parse_zone(zone) for zone in distinct]
    k = len(distinct)
    between = np.zeros((k + 1, k + 1))  # last row/column: no zone (the station)
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            cost = penalties.different_zone
            if parsed[a] is not None and parsed[b] is not None:
                if parsed[a][0] != parsed[b][0]:
                    cost += penalties.different_region
                elif parsed[a][1] != parsed[b][1] and parsed[a][2] != parsed[b][2]:
                    cost += penalties.both_parts
            between[a, b] = cost
    codes = np.array([index[zone] if zone else k for zone in zones])
    return between[codes[:, None], codes[None, :]]


def zone_aware_route(
    route: Route,
    penalties: ZonePenalties = ZonePenalties(),
    rounds: int = 20,
    seed: int = 0,
) -> list[int]:
    """Shortest route when leaving a zone, sub-area or region costs extra."""
    cost = route.times + penalty_matrix(stop_zones(route), penalties)
    return tsp.solve(cost, rounds=rounds, seed=seed).tolist()
