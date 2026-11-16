"""Learned model of the order in which drivers visit delivery zones.

Three small choice models are fitted on historical routes:

* which zone a driver goes to next, given the current zone and the zones left;
* which zone a route starts in;
* which zone a route ends in.

Each is a conditional logit (a softmax over the candidate zones). The inputs
describe a candidate zone: how often drivers from the same station made that
move before, how the zone codes relate, and how far away the zone is.

To sequence a new route, the learned preferences are turned into extra cost on
the travel-time matrix (unlikely moves cost more) and the route solver finds the
shortest route under those costs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import tsp, zones
from .data import Route

N_PAIR_FEATURES = 12
N_END_FEATURES = 4


# ------------------------------------------------------------------ zone view


@dataclass
class ZoneView:
    """A route reduced to its zones."""

    route_id: str
    station_code: str
    zones: list[str]
    stop_zone: np.ndarray  # zone number of every stop; -1 for the station
    mean_time: np.ndarray  # [a, b] mean travel time from stops of a to stops of b
    min_time: np.ndarray  # [a, b] shortest hop from a stop of a to a stop of b
    from_station: np.ndarray  # mean travel time station -> zone
    to_station: np.ndarray
    order: list[int] | None  # zones in the order the driver first reached them


def zone_view(route: Route) -> ZoneView:
    labels = [zone or "?" for zone in zones.stop_zones(route)[1:]]
    names = sorted(set(labels))
    number = {name: i for i, name in enumerate(names)}
    stop_zone = np.array([-1] + [number[label] for label in labels])
    k = len(names)

    member = np.zeros((k, route.n_stops))
    member[stop_zone[1:], np.arange(1, route.n_stops)] = 1.0
    size = member.sum(axis=1)
    mean_time = member @ route.times @ member.T / np.outer(size, size)
    members = [np.flatnonzero(stop_zone == z) for z in range(k)]
    min_time = np.zeros((k, k))
    for a in range(k):
        rows = route.times[members[a]]
        for b in range(k):
            if a != b:
                min_time[a, b] = rows[:, members[b]].min()

    order = None
    if route.actual is not None:
        order = list(dict.fromkeys(int(stop_zone[stop]) for stop in route.actual[1:]))

    return ZoneView(
        route_id=route.id,
        station_code=route.station_code,
        zones=names,
        stop_zone=stop_zone,
        mean_time=mean_time,
        min_time=min_time,
        from_station=member @ route.times[0] / size,
        to_station=member @ route.times[:, 0] / size,
        order=order,
    )


# -------------------------------------------------------------------- history


@dataclass
class History:
    """How often drivers of each station made each zone move in past routes."""

    pair: dict = field(default_factory=lambda: defaultdict(Counter))  # (from, to)
    first: dict = field(default_factory=lambda: defaultdict(Counter))
    last: dict = field(default_factory=lambda: defaultdict(Counter))

    def add(self, view: ZoneView) -> None:
        if not view.order:
            return
        names = [view.zones[z] for z in view.order]
        self.first[view.station_code][names[0]] += 1
        self.last[view.station_code][names[-1]] += 1
        self.pair[view.station_code].update(zip(names, names[1:]))

    def counts(self, view: ZoneView, exclude_own: bool):
        """Counts for the zones of ``view``: pair [a, b], first [b], last [b].

        ``exclude_own`` removes the route's own contribution, so that a route
        used for fitting does not see its own answer.
        """
        k = len(view.zones)
        pair_counts = self.pair.get(view.station_code, {})
        first_counts = self.first.get(view.station_code, {})
        last_counts = self.last.get(view.station_code, {})
        pair = np.zeros((k, k))
        if pair_counts:
            for a, name_a in enumerate(view.zones):
                for b, name_b in enumerate(view.zones):
                    if a != b:
                        pair[a, b] = pair_counts.get((name_a, name_b), 0)
        first = np.array([first_counts.get(name, 0) for name in view.zones], dtype=float)
        last = np.array([last_counts.get(name, 0) for name in view.zones], dtype=float)
        if exclude_own and view.order:
            order = view.order
            pair[order[:-1], order[1:]] -= 1
            first[order[0]] -= 1
            last[order[-1]] -= 1
        return pair, first, last


# ------------------------------------------------------------------- features


def _rank(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """1 for the smallest value along ``axis``, 2 for the next, ..."""
    return values.argsort(axis=axis).argsort(axis=axis) + 1.0


def pair_features(view: ZoneView, pair_counts: np.ndarray) -> np.ndarray:
    """Features of moving from zone a to zone b, shape [k, k, N_PAIR_FEATURES]."""
    k = len(view.zones)
    parsed = [zones.parse_zone(name) for name in view.zones]
    features = np.zeros((k, k, N_PAIR_FEATURES))
    features[:, :, 0] = np.log1p(pair_counts)
    features[:, :, 1] = np.log1p(pair_counts.T)
    for a in range(k):
        for b in range(k):
            if a == b or parsed[a] is None or parsed[b] is None:
                continue
            (region_a, sub_a, block_a), (region_b, sub_b, block_b) = parsed[a], parsed[b]
            if region_a != region_b:
                continue
            sub_step = int(sub_b) - int(sub_a)
            block_step = ord(block_b[-1]) - ord(block_a[-1]) if len(block_a) == len(block_b) else 99
            if sub_step == 0:  # only the block letter changes
                features[a, b, 2] = 1
                features[a, b, 5] = abs(block_step) == 1
                features[a, b, 11] = block_step > 0
            elif block_a == block_b:  # only the sub-area changes
                features[a, b, 3] = 1
                features[a, b, 6] = abs(sub_step) == 1
                features[a, b, 10] = sub_step > 0
            else:
                features[a, b, 4] = 1
    off_diagonal = ~np.eye(k, dtype=bool)
    scale = np.median(view.mean_time[off_diagonal]) if k > 1 else 1.0
    features[:, :, 7] = view.mean_time / max(scale, 1.0)
    nearness = np.where(off_diagonal, view.min_time, np.inf)
    rank = _rank(nearness, axis=1)
    features[:, :, 8] = np.log(rank)
    features[:, :, 9] = rank == 1
    return features


def end_features(times: np.ndarray, primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Features of a zone as the first (or last) of a route, shape [k, N_END_FEATURES].

    ``primary`` counts how often the zone played this role before, ``secondary``
    how often it played the opposite role.
    """
    k = len(times)
    features = np.zeros((k, N_END_FEATURES))
    features[:, 0] = np.log1p(primary)
    features[:, 1] = np.log1p(secondary)
    features[:, 2] = (_rank(times) - 1) / max(k - 1, 1)
    features[:, 3] = times / max(times.mean(), 1.0)
    return features


# ------------------------------------------------------------------- fitting


def fit_choice(rows: np.ndarray, group_sizes: np.ndarray, steps: int = 400, l2: float = 1e-4) -> np.ndarray:
    """Fit a conditional logit. The chosen option is the first row of each group."""
    starts = np.concatenate(([0], np.cumsum(group_sizes)[:-1]))
    n_groups = len(group_sizes)
    chosen = np.zeros(len(rows))
    chosen[starts] = 1.0
    weights = np.zeros(rows.shape[1])
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    for step in range(1, steps + 1):  # Adam on the full data; the problem is convex
        utility = rows @ weights
        top = np.repeat(np.maximum.reduceat(utility, starts), group_sizes)
        expo = np.exp(utility - top)
        prob = expo / np.repeat(np.add.reduceat(expo, starts), group_sizes)
        gradient = rows.T @ (prob - chosen) / n_groups + l2 * weights
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient**2
        weights -= 0.05 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    return weights


def _log_softmax(utility: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = utility - utility.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


@dataclass
class ZoneModel:
    history: History
    w_pair: np.ndarray
    w_start: np.ndarray
    w_end: np.ndarray
    preference_weight: float = 300.0  # seconds per unit of learned cost
    zone_change: float = 600.0  # seconds for leaving any zone

    @classmethod
    def fit(cls, views: list[ZoneView], **settings) -> "ZoneModel":
        history = History()
        for view in views:
            history.add(view)

        pair_rows, pair_sizes = [], []
        start_rows, start_sizes = [], []
        end_rows, end_sizes = [], []
        for view in views:
            order = view.order
            if not order or len(order) < 2:
                continue
            pair, first, last = history.counts(view, exclude_own=True)
            features = pair_features(view, pair)
            for t in range(len(order) - 2):  # with one zone left there is no choice
                pair_rows.append(features[order[t], order[t + 1 :]])
                pair_sizes.append(len(order) - t - 1)
            others = [z for z in range(len(view.zones)) if z != order[0]]
            start_rows.append(end_features(view.from_station, first, last)[[order[0]] + others])
            start_sizes.append(len(view.zones))
            others = [z for z in range(len(view.zones)) if z != order[-1]]
            end_rows.append(end_features(view.to_station, last, first)[[order[-1]] + others])
            end_sizes.append(len(view.zones))

        return cls(
            history=history,
            w_pair=fit_choice(np.concatenate(pair_rows), np.array(pair_sizes)),
            w_start=fit_choice(np.concatenate(start_rows), np.array(start_sizes)),
            w_end=fit_choice(np.concatenate(end_rows), np.array(end_sizes)),
            **settings,
        )

    def zone_costs(self, view: ZoneView, exclude_own: bool = False):
        """Learned cost (negative log-probability) of each zone move, start and end."""
        k = len(view.zones)
        pair, first, last = self.history.counts(view, exclude_own)
        utility = pair_features(view, pair) @ self.w_pair
        utility[np.eye(k, dtype=bool)] = -np.inf
        move = -_log_softmax(utility, axis=1) if k > 1 else np.zeros((k, k))
        move[np.eye(k, dtype=bool)] = 0.0
        start = -_log_softmax(end_features(view.from_station, first, last) @ self.w_start)
        end = -_log_softmax(end_features(view.to_station, last, first) @ self.w_end)
        return move, start, end

    def cost_matrix(self, route: Route, view: ZoneView | None = None) -> np.ndarray:
        """Travel times plus the learned cost of every zone move."""
        view = view or zone_view(route)
        k = len(view.zones)
        move, start, end = self.zone_costs(view)
        between = np.zeros((k + 1, k + 1))  # last row/column stands for the station
        between[:k, :k] = self.preference_weight * move + self.zone_change * (1 - np.eye(k))
        between[k, :k] = self.preference_weight * start
        between[:k, k] = self.preference_weight * end
        codes = np.where(view.stop_zone < 0, k, view.stop_zone)
        return route.times + between[codes[:, None], codes[None, :]]

    def predict(self, route: Route, rounds: int = 20, seed: int = 0) -> list[int]:
        """Proposed stop sequence for a route, starting at the station."""
        return tsp.solve(self.cost_matrix(route), rounds=rounds, seed=seed).tolist()
