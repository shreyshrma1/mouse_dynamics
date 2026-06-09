"""
train_from_existing_shen_scroll.py

Trains a One-Class SVM for a single user from their bank collection sessions.
All feature sets (holistic, scroll, more_scroll, dir_scroll) are always used.

Defaults:
    data_dir = bank_collection/bank-data
    save_dir = checkpoints_shen_scroll_bank

Usage:
    python shen_model/train_from_existing_shen_scroll.py
    python shen_model/train_from_existing_shen_scroll.py --top_n 15
    python shen_model/train_from_existing_shen_scroll.py
        --data_dir collected_data --save_dir checkpoints_shen_scroll_collected
"""

print("importing sys and os")
import sys
import os
print("importing joblib")
import joblib
print("importing numpy")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("importing extract_features_scroll")
from measurements.extract_features_scroll import extract_session_features, MORE_SCROLL_COLS, DIR_SCROLL_COLS
print("done importing extract_features_scroll")
print("importing sklearn")
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
print("done importing sklearn")

# Defaults (original behaviour)
DEFAULT_DATA_DIR = "bank_collection/bank-data"
DEFAULT_SAVE_DIR = "checkpoints_shen_scroll_bank"
WINDOW_SIZE = 5
NU = 0.06
GAMMA = "scale"
HELD_OUT_FRAC = 0.25

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
    "scroll_count",
    "scroll_rate",
    "scroll_ratio",
    "scroll_up_ratio",
    "scroll_dur_mean",
    "scroll_dur_std",
    "scroll_burst_count",
    "scroll_burst_dur_mean",
    "scroll_burst_len_mean",
]

ALL_FEATURE_COLS = HOLISTIC_COLS + SCROLL_COLS  # extended by MORE_SCROLL_COLS + DIR_SCROLL_COLS at runtime

# Top-N ranked features from permutation importance (feat_importance.py)
RANKED_FEATURES = [
    "num_critical_points",
    "num_points",
    "sum_of_angles",
    "scroll_rate",
    "scroll_burst_len_mean",
    "sd_omega",
    "scroll_count",
    "straightness",
    "max_vx",
    "sd_curv",
    "scroll_ratio",
    "max_a",
    "mean_omega",
    "min_vx",
    "min_vy",
]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"Root directory containing per-user session folders "
                             f"(default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR,
                        help=f"Directory to save model checkpoints "
                             f"(default: {DEFAULT_SAVE_DIR})")
    parser.add_argument("--top_n", type=int, default=None,
                        help="Use only the top-N ranked features (1-15). "
                             "Omit to use all features.")
    return parser.parse_args()


def get_session_files(user_dir):
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def extract_all_windows(session_files, user_id, feature_cols):
    all_vecs = []
    for path in session_files:
        print(f"  Processing {os.path.basename(path)}...")
        try:
            df = extract_session_features(path, user_id, window_size=WINDOW_SIZE,
                                          more_scroll=True, dir_scroll=True)
            if df.empty or not all(c in df.columns for c in feature_cols):
                continue
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
            if len(df) == 0:
                continue
            for _, grp in df.groupby("session"):
                rows = grp[feature_cols].values
                if len(rows) >= 1:
                    all_vecs.append(rows.mean(axis=0))
        except Exception as e:
            print(f"  [!] {os.path.basename(path)}: {e}")
    return all_vecs


def find_reference(train_samples):
    n = len(train_samples)
    print(f"Running find_reference on {n} samples...")
    mean_dists = np.zeros(n)
    for i in range(n):
        dists = np.sum(np.abs(train_samples - train_samples[i]), axis=1)
        mean_dists[i] = dists.sum() / max(n - 1, 1)
    print("Done.")
    return train_samples[np.argmin(mean_dists)]


def main():
    args = parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir

    print(f"[Config] data_dir={data_dir}  save_dir={save_dir}")

    if args.top_n is not None:
        top_n = max(1, min(args.top_n, len(RANKED_FEATURES)))
        feature_cols = RANKED_FEATURES[:top_n]
        print(f"Using top-{top_n} features: {feature_cols}")
    else:
        top_n = None
        feature_cols = ALL_FEATURE_COLS + MORE_SCROLL_COLS + DIR_SCROLL_COLS
        print(f"Using all {len(feature_cols)} features")

    user_id = input("Enter user ID: ").strip()

    user_dir = os.path.join(data_dir, user_id)
    if not os.path.isdir(user_dir):
        print(f"No data directory found at {user_dir}")
        sys.exit(1)

    session_files = get_session_files(user_dir)
    print(f"Found {len(session_files)} session files for {user_id}")

    all_vecs = extract_all_windows(session_files, user_id, feature_cols)
    if len(all_vecs) < 8:
        print(f"Not enough windows ({len(all_vecs)}) to train — collect more data")
        sys.exit(1)

    n_total = len(all_vecs)
    n_test = max(1, int(n_total * HELD_OUT_FRAC))
    n_train = n_total - n_test

    train_samples = np.array(all_vecs[:n_train])
    test_samples = np.array(all_vecs[n_train:])

    print(f"Total windows: {n_total}  |  Train: {n_train}  |  Held-out: {n_test}")
    print(f"Feature vector: {len(feature_cols)} features")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_samples)

    reference = find_reference(train_scaled)
    train_dists = np.abs(train_scaled - reference)
    dist_mean = train_dists.mean(axis=0)
    dist_std = train_dists.std(axis=0)
    std_safe = np.where(dist_std < 1e-9, 1.0, dist_std)
    train_norm = (train_dists - dist_mean) / std_safe

    model = OneClassSVM(kernel="rbf", nu=NU, gamma=GAMMA)
    model.fit(train_norm)

    scores = model.decision_function(train_norm)
    print(f"Train scores: min={scores.min():.4f}, mean={scores.mean():.4f}, max={scores.max():.4f}")

    save_path = os.path.join(save_dir, user_id)
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(model, os.path.join(save_path, "model.pkl"))
    joblib.dump({
        "reference": reference,
        "dist_mean": dist_mean,
        "dist_std": dist_std,
        "scaler": scaler,
        "test_samples": test_samples,
        "feature_cols": feature_cols,
        "top_n": top_n,
        "n_train": n_train,
        "n_test": n_test,
        "nu": NU,
        "gamma": GAMMA,
        "window_size": WINDOW_SIZE,
    }, os.path.join(save_path, "state.pkl"))

    print(f"Model saved to {save_path}/")


if __name__ == "__main__":
    main()