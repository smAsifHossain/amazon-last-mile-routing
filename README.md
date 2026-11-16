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

### Official evaluation routes

The model was fitted on all 6,112 training routes and scored on the 3,052
evaluation routes released with the challenge, using the challenge's metric.
Lower is better.

| Method | Mean score |
|---|---|
| Shortest route | 0.0882 |
| Zone-aware, fixed penalties | 0.0514 |
| **Learned zone order** | **0.0251** |
| Driver's own zone order (limit of this approach) | 0.0074 |

The learned model is closer to the driver than the shortest route on 97% of the
routes. It starts in the driver's first zone on 61% of routes and reproduces 77%
of the drivers' zone-to-zone moves.

Reproduce with `python scripts/evaluate.py` (about 35 minutes on 4 cores).

For reference, the 2021 challenge was won with a score of 0.0248, and the
runner-up scored 0.0353. The comparison needs care:

- The challenge is closed, so this score is self-reported. It was not run on the
  organisers' machines or under their limits on training and routing time.
- This project was built after the challenge, with the winning teams' published
  insight that delivery zones drive driver behaviour.
- Scoring uses this project's port of the official scorer, which is tested
  against the official code.

### Held-out training routes

During development the model was validated on 1,186 training routes held out of
fitting (`python scripts/validate.py`): 0.0256 for the learned model against
0.0916 for the shortest route. The official result above agrees with it.

## Demo

`docs/` holds a static web app with three views:

- **Home**: what the project is, what it does and how.
- **Results**: scores on the 3,052 official evaluation routes, by method, by city
  and route by route.
- **Explore routes**: 60 evaluation routes on a map, with the driver's sequence
  beside the shortest route or the model's route. Routes can be played back stop
  by stop, and a strip shows where the predicted zone order departs from the
  driver's.

```bash
python -m http.server 8765 --directory docs
```

Then open http://localhost:8765. Its data files are written by
`python scripts/evaluate.py`.

## Status

- [x] Scorer, verified against the official challenge scorer
- [x] Data download and conversion that fits in little memory
- [x] Baselines: random order, nearest neighbour, shortest route, fixed zone penalties
- [x] Learned zone-order model, validated on held-out training routes
- [x] Score on the 3,052 official evaluation routes
- [ ] Better zone-order prediction (the first zone is right on 61% of routes)
- [x] Interactive demo (local)
- [ ] Demo hosted online

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
python scripts/prepare_data.py eval
```

It contains 6,112 training routes and 3,052 evaluation routes from five US
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
