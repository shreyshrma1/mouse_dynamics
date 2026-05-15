"""
extract_features_action.py

Extends the base feature extractor to return window objects containing:
  - holistic_vec: 42-dim window vector
      [39-dim global averaged feature vector, prop_MM, prop_PC, prop_DD]
  - curves_by_type: dict mapping action type to list of (speed, accel) tuples
      {
        ACTION_MM: [(v, a), (v, a), ...],
        ACTION_PC: [(v, a), ...],
        ACTION_DD: [(v, a), ...],
      }

This compact version keeps the original global 39-dim holistic average and
adds only 3 action-composition features. It does NOT split the holistic vector
into separate MM/PC/DD blocks.

Usage:
    from measurements.extract_features_action import extract_procedural_windows
    windows = extract_procedural_windows(session_files, user_id, window_size=50)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_sess import (
    load_session,
    segment_actions,
    compute_time_series,
    extract_action_features,
    ACTION_MM,
    ACTION_PC,
    ACTION_DD,
)

FEATURE_COLS = [
    "type_of_action", "traveled_distance_pixel", "elapsed_time",
    "direction_of_movement", "straightness", "num_points", "sum_of_angles",
    "mean_curv", "sd_curv", "max_curv", "min_curv",
    "mean_omega", "sd_omega", "max_omega", "min_omega",
    "largest_deviation", "dist_end_to_end_line", "num_critical_points",
    "mean_vx", "sd_vx", "max_vx", "min_vx",
    "mean_vy", "sd_vy", "max_vy", "min_vy",
    "mean_v", "sd_v", "max_v", "min_v",
    "mean_a", "sd_a", "max_a", "min_a",
    "mean_jerk", "sd_jerk", "max_jerk", "min_jerk", "a_beg_time"
]

ACTION_TYPES = [ACTION_MM, ACTION_PC, ACTION_DD]
N_BASE_FEATURES = len(FEATURE_COLS)      # 39 with current list
N_PROP_FEATURES = len(ACTION_TYPES)      # prop_MM, prop_PC, prop_DD
HOLISTIC_DIM = N_BASE_FEATURES + N_PROP_FEATURES  # 42


def extract_procedural_windows(session_files, user_id, window_size=50):
    """
    Extract window objects from a list of session files.

    Each window object contains:
      - holistic_vec: (42,) array
            [global mean of per-action 39-dim vectors,
             prop_MM, prop_PC, prop_DD]
      - curves_by_type: dict mapping action type int to list of
                        (speed_array, accel_array) tuples

    Returns a list of window dicts in session order.
    """
    all_windows = []

    for path in session_files:
        try:
            df = load_session(path)
            actions = segment_actions(df)
        except Exception as e:
            print(f"  [!] {os.path.basename(path)}: {e}")
            continue

        for start in range(0, len(actions), window_size):
            window_actions = actions[start: start + window_size]

            if len(window_actions) < 5:
                continue

            holistic_rows = []
            curves_by_type = {ACTION_MM: [], ACTION_PC: [], ACTION_DD: []}
            action_counts = {ACTION_MM: 0, ACTION_PC: 0, ACTION_DD: 0}

            for action in window_actions:
                ts = compute_time_series(action["events"])
                feat = extract_action_features(action["type"], ts)

                if feat is None:
                    continue

                # Keep the original global 39-dim per-action representation.
                # This intentionally includes type_of_action, matching the
                # previous/well-performing baseline representation.
                row = np.array([feat[col] for col in FEATURE_COLS], dtype=float)
                holistic_rows.append(row)

                action_type = action["type"]

                if action_type in action_counts:
                    action_counts[action_type] += 1

                if action_type in curves_by_type:
                    speed = ts["v"].copy().astype(np.float64)
                    accel = ts["a"].copy().astype(np.float64)
                    curves_by_type[action_type].append((speed, accel))

            if len(holistic_rows) == 0:
                continue

            # Skip windows with no procedural curves for any tracked action type.
            if not any(len(v) > 0 for v in curves_by_type.values()):
                continue

            global_mean = np.mean(holistic_rows, axis=0)

            total_count = sum(action_counts.values())
            if total_count > 0:
                props = np.array([
                    action_counts[ACTION_MM] / total_count,
                    action_counts[ACTION_PC] / total_count,
                    action_counts[ACTION_DD] / total_count,
                ], dtype=np.float64)
            else:
                props = np.zeros(N_PROP_FEATURES, dtype=np.float64)

            holistic_vec = np.concatenate([global_mean, props])

            if holistic_vec.shape[0] != HOLISTIC_DIM:
                raise ValueError(
                    f"Expected holistic dim {HOLISTIC_DIM}, "
                    f"got {holistic_vec.shape[0]}"
                )

            all_windows.append({
                "holistic_vec": holistic_vec,
                "curves_by_type": curves_by_type,
            })

    return all_windows