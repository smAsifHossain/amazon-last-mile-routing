# amazon-last-mile-routing

Learning the stop sequences that experienced delivery drivers actually follow, using
the public dataset of the 2021 Amazon Last Mile Routing Research Challenge.

## The problem

A shortest-route solver gives the sequence that minimises travel time. Experienced
drivers often deviate from it, because they know things the map does not: where
parking is easy, which streets to avoid, which customers are in at what time. The
challenge asks for a model that, given a new route, predicts the sequence a skilled
driver would follow.

A proposed sequence is scored by how far it deviates from the driver's actual
sequence. Lower is better, and 0 is a perfect match.

## Approach

A diagnostic on held-out routes showed where the difficulty lies. Every drop-off
belongs to a delivery zone, and drivers serve one zone after another. If the order
of the zones is known, even a simple stop order inside each zone scores about
0.007. Almost all of the problem is predicting the zone order.

The model therefore works in two stages:

1. **Zone order (learned).** Three small choice models are fitted on past routes:
   which zone a driver goes to next, which zone a route starts in, and which it
   ends in. They use how often drivers from the same station made each move
   before, how the zone codes relate, and how far apart the zones are. The zone
   order with the lowest total learned cost is chosen.
2. **Stop order (optimised).** A route solver builds the shortest stop sequence
   that serves the zones in that order.

## Results

Mean score on 1,186 training routes that were held out: the model saw neither
their sequences nor their zone history. Lower is better.

| Method | All routes | High-quality routes (526) |
|---|---|---|
| Shortest route | 0.0916 | 0.0863 |
| Zone-aware, fixed penalties | 0.0550 | 0.0528 |
| **Learned zone order** | **0.0256** | **0.0224** |
| Driver's own zone order (limit of this approach) | 0.0073 | 0.0072 |

Reproduce with `python scripts/validate.py` (about 8 minutes on 4 cores).

These are not leaderboard numbers. The challenge leaderboard uses a separate set
of 3,072 evaluation routes, which this project has not been scored on yet.

## Status

- [x] Scorer, verified against the official challenge scorer
- [x] Data download and conversion that fits in little memory
- [x] Baselines: random order, nearest neighbour, shortest route, fixed zone penalties
- [x] Learned zone-order model, validated on held-out training routes
- [ ] Score on the 3,072 official evaluation routes
- [ ] Better zone-order prediction (first zone is right about two times in three)
- [ ] Interactive demo

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"
pytest
```

## Data

The dataset (about 3.3 GB) is not stored in this repository. Download it from the
[AWS Registry of Open Data](https://registry.opendata.aws/amazon-last-mile-challenges):

```bash
python scripts/download_data.py --list            # show files and sizes
python scripts/download_data.py --subset sample   # a handful of routes (~5 MB)
python scripts/download_data.py --subset train    # 6,112 training routes (~2.3 GB)
python scripts/download_data.py                   # everything
```

Then convert the raw files into one compact file per route. This reads one route
at a time, so it runs on a machine with a few gigabytes of memory:

```bash
python scripts/prepare_data.py train
```

It contains 6,112 training routes and 3,072 evaluation routes from five US
metropolitan areas, with stop locations, package details, travel times between
stops and the sequence each driver followed.

## Layout

```
src/lastmile/        project code
  scoring.py         challenge metric
  data.py            routes, raw-file reader, per-route files
  tsp.py             route solver (iterated local search)
  baselines.py       methods that use travel time only
  zones.py           zone codes and the fixed-penalty method
  model.py           learned zone-order model
scripts/             command-line scripts
tests/               tests
third_party/rc_cli/  reference copy of the official scorer (MIT licence)
data/                downloaded dataset (not in git)
```

## Licence and attribution

Code in this repository is released under the MIT licence (see `LICENSE`).

The dataset is provided by Amazon under the Creative Commons
Attribution-NonCommercial 4.0 licence and may not be used commercially:

> Merchán, D., et al. (2022). 2021 Amazon Last Mile Routing Research Challenge:
> Data Set. *Transportation Science*. https://doi.org/10.1287/trsc.2022.1173

This project is independent and is not affiliated with or endorsed by Amazon.
