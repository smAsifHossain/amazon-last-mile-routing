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

## Status

Work in progress.

- [x] Project setup, data download script
- [x] Scorer, verified against the official challenge scorer
- [x] Data loading
- [x] Simple baselines (random order, nearest neighbour)
- [x] Shortest-route solver (iterated local search, 2-opt and segment moves)
- [x] Zone-aware heuristic with hand-set penalties
- [ ] Zone-to-zone patterns learned from the 6,112 training routes
- [ ] Evaluation on the 3,072 held-out routes
- [ ] Interactive demo

### Results so far

Mean score on the 13 sample routes that ship with the training data (lower is
better). This is a development check only: the sample is too small to compare with
the challenge leaderboard, and the zone penalties were chosen on these same routes.

| Method | Score |
|---|---|
| Random order | 0.888 |
| Nearest neighbour | 0.122 |
| Shortest route | 0.086 |
| Zone-aware shortest route | 0.043 |

Reproduce with `python scripts/run_baselines.py`. The solver runs under a time
limit, so scores vary slightly between runs and machines.

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
python scripts/download_data.py                   # everything
```

It contains 6,112 training routes and 3,072 evaluation routes from five US
metropolitan areas, with stop locations, package details, travel times between
stops and the sequence each driver followed.

## Layout

```
src/lastmile/        project code
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
