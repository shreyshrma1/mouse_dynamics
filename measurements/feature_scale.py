import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_scroll import extract_session_features

DATA_DIR = "balabit_dataset/training_files"
HOLISTIC_COLS = [
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
SCROLL_COLS = [
    "scroll_count","scroll_rate","scroll_ratio","scroll_up_ratio",
    "scroll_dur_mean","scroll_dur_std",
    "scroll_burst_count","scroll_burst_dur_mean","scroll_burst_len_mean",
]
FEATURE_COLS = HOLISTIC_COLS + SCROLL_COLS

# Load one user
user = "user7"
user_dir = os.path.join(DATA_DIR, user)
files = sorted([os.path.join(user_dir, f) for f in os.listdir(user_dir)
                if os.path.isfile(os.path.join(user_dir, f))])

all_vecs = []
for path in files[:3]:  # just first 3 sessions
    df = extract_session_features(path, user, window_size=50)
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=FEATURE_COLS)
    for _, grp in df.groupby("session"):
        rows = grp[FEATURE_COLS].values
        if len(rows) >= 1:
            all_vecs.append(rows.mean(axis=0))

X = np.array(all_vecs)
print(f"Shape: {X.shape}")
print()
print(f"{'Feature':<35} {'Mean':>12} {'Std':>12} {'Max':>12}")
print("-" * 75)
for i, col in enumerate(FEATURE_COLS):
    print(f"{col:<35} {X[:,i].mean():>12.2f} {X[:,i].std():>12.2f} {np.abs(X[:,i]).max():>12.2f}")