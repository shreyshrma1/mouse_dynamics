"""
extract_features_procedural.py

Extends the base feature extractor to return window objects containing:
  - holistic_vec: 39-dim averaged feature vector for the window
  - curves: list of (speed_array, accel_array) tuples, one per action

Each window object is a dict:
  {
    "holistic_vec": np.array (39,),
    "curves": [(v1, a1), (v2, a2), ..., (v_n, a_n)]
  }

Usage:
    from measurements.extract_features_procedural import extract_procedural_windows
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
)

FEATURE_COLS = [
    "type_of_action","traveled_distance_pixel","elapsed_time",
    "direction_of_movement","straightness","num_points","sum_of_angles",
    "mean_curv","sd_curv","max_curv","min_curv",
    "mean_omega","sd_omega","max_omega","min_omega",
    "largest_deviation","dist_end_to_end_line","num_critical_points",
    "mean_vx","sd_vx","max_vx","min_vx",
    "mean_vy","sd_vy","max_vy","min_vy",
    "mean_v","sd_v","max_v","min_v",
    "mean_a","sd_a","max_a","min_a",
    "mean_jerk","sd_jerk","max_jerk","min_jerk","a_beg_time"
]

FEATURE_COL_ORDER = FEATURE_COLS


def extract_procedural_windows(session_files, user_id, window_size=50):
    """
    Extract window objects from a list of session files.

    Each window object contains:
      - holistic_vec: (39,) array — mean of per-action 39-dim vectors
      - curves: list of (speed, accel) tuples, one per action in the window
                speed and accel are 1-D numpy arrays of variable length

    Returns a list of window dicts in session order.
    """
    all_windows = []

    for path in session_files:
        try:
            df      = load_session(path)
            actions = segment_actions(df)
        except Exception as e:
            print(f"  [!] {os.path.basename(path)}: {e}")
            continue

        # slice actions into fixed-size windows
        for start in range(0, len(actions), window_size):
            window_actions = actions[start: start + window_size]

            if len(window_actions) < 5:
                continue  # skip short windows

            holistic_rows = []
            curves        = []

            for action in window_actions:
                ts   = compute_time_series(action["events"])
                feat = extract_action_features(action["type"], ts)

                if feat is None:
                    continue

                # 39-dim holistic vector for this action
                row = np.array([feat[col] for col in FEATURE_COL_ORDER], dtype=float)
                holistic_rows.append(row)

                # raw speed and acceleration curves (variable length)
                speed = ts["v"].copy()   # tangential speed
                accel = ts["a"].copy()   # tangential acceleration
                curves.append((speed, accel))

            if len(holistic_rows) == 0 or len(curves) == 0:
                continue

            holistic_vec = np.mean(holistic_rows, axis=0)  # (39,)

            all_windows.append({
                "holistic_vec": holistic_vec,
                "curves":       curves,
            })

    return all_windows