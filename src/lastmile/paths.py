"""Locations of the project's data files."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"  # one compact file per route

TRAIN_DIR = RAW_DIR / "almrrc2021-data-training"
EVAL_DIR = RAW_DIR / "almrrc2021-data-evaluation"

# Sample: 13 routes shipped inside the training folder, with actual sequences.
SAMPLE_ROUTES = TRAIN_DIR / "model_apply_inputs" / "new_route_data.json"
SAMPLE_PACKAGES = TRAIN_DIR / "model_apply_inputs" / "new_package_data.json"
SAMPLE_TRAVEL_TIMES = TRAIN_DIR / "model_apply_inputs" / "new_travel_times.json"
SAMPLE_ACTUAL = TRAIN_DIR / "model_score_inputs" / "new_actual_sequences.json"
SAMPLE_INVALID_SCORES = TRAIN_DIR / "model_score_inputs" / "new_invalid_sequence_scores.json"

# Training set: 6,112 historical routes with the sequence the driver followed.
TRAIN_ROUTES = TRAIN_DIR / "model_build_inputs" / "route_data.json"
TRAIN_PACKAGES = TRAIN_DIR / "model_build_inputs" / "package_data.json"
TRAIN_TRAVEL_TIMES = TRAIN_DIR / "model_build_inputs" / "travel_times.json"
TRAIN_ACTUAL = TRAIN_DIR / "model_build_inputs" / "actual_sequences.json"
TRAIN_INVALID_SCORES = TRAIN_DIR / "model_build_inputs" / "invalid_sequence_scores.json"

# Evaluation set: 3,072 routes used for the challenge leaderboard.
EVAL_ROUTES = EVAL_DIR / "model_apply_inputs" / "eval_route_data.json"
EVAL_PACKAGES = EVAL_DIR / "model_apply_inputs" / "eval_package_data.json"
EVAL_TRAVEL_TIMES = EVAL_DIR / "model_apply_inputs" / "eval_travel_times.json"
EVAL_ACTUAL = EVAL_DIR / "model_score_inputs" / "eval_actual_sequences.json"
EVAL_INVALID_SCORES = EVAL_DIR / "model_score_inputs" / "eval_invalid_sequence_scores.json"
