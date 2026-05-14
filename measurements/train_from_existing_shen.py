"""
train_from_existing_shen.py

Trains the Shen et al. pipeline on a user's existing collected data.

Saves model components to checkpoints_shen/<user_id>/ for later scoring.

Usage:
    python measurements/train_from_existing_shen.py
"""

import sys
import os
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_sess import extract_session_features
from sklearn.svm import OneClassSVM

DATA_DIR    = "collected_data"
SAVE_DIR    = "checkpoints_shen"
WINDOW_SIZE = 50
NU          = 0.06
GAMMA       = "scale"

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


def get_session_files(user_dir):
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def extract_all_windows(session_files, user_id):
    all_vecs = []
    for path in session_files:
        try:
            df = extract_session_features(path, user_id, window_size=WINDOW_SIZE)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
            if len(df) == 0:
                continue
            for _, grp in df.groupby("session"):
                rows = grp[FEATURE_COLS].values
                if len(rows) >= 1:
                    all_vecs.append(rows.mean(axis=0))
        except Exception as e:
            print(f"  [!] {os.path.basename(path)}: {e}")
    return all_vecs


def find_reference(train_samples):
    n = len(train_samples)
    mean_dists = np.zeros(n)
    for i in range(n):
        dists = np.sum(np.abs(train_samples - train_samples[i]), axis=1)
        mean_dists[i] = dists.sum() / max(n - 1, 1)
    return train_samples[np.argmin(mean_dists)]


def distance_vectors(samples, reference):
    return np.abs(np.atleast_2d(samples) - reference)


def normalize(dist_vecs, mean, std):
    std_safe = np.where(std < 1e-9, 1.0, std)
    return (dist_vecs - mean) / std_safe


def main():
    user_id = input("Enter user ID: ").strip()

    user_dir = os.path.join(DATA_DIR, user_id)
    if not os.path.isdir(user_dir):
        print(f"No data directory found at {user_dir}")
        sys.exit(1)

    session_files = get_session_files(user_dir)
    print(f"Found {len(session_files)} session files for {user_id}")

    all_vecs = extract_all_windows(session_files, user_id)
    if len(all_vecs) < 5:
        print(f"Not enough windows ({len(all_vecs)}) to train — collect more data")
        sys.exit(1)

    train_samples = np.array(all_vecs)
    print(f"Extracted {len(train_samples)} windows")

    # Shen pipeline
    reference   = find_reference(train_samples)
    train_dists = distance_vectors(train_samples, reference)
    dist_mean   = train_dists.mean(axis=0)
    dist_std    = train_dists.std(axis=0)
    train_norm  = normalize(train_dists, dist_mean, dist_std)

    model = OneClassSVM(kernel="rbf", nu=NU, gamma=GAMMA)
    model.fit(train_norm)

    scores = model.decision_function(train_norm)
    print(f"Train scores: min={scores.min():.4f}, mean={scores.mean():.4f}, max={scores.max():.4f}")

    # Save
    save_path = os.path.join(SAVE_DIR, user_id)
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(model,     os.path.join(save_path, "model.pkl"))
    joblib.dump({
        "reference":  reference,
        "dist_mean":  dist_mean,
        "dist_std":   dist_std,
        "n_windows":  len(train_samples),
        "nu":         NU,
        "gamma":      GAMMA,
        "window_size":WINDOW_SIZE,
    }, os.path.join(save_path, "state.pkl"))

    print(f"Model saved to {save_path}/")
    print(f"Training complete — {len(train_samples)} windows used")


if __name__ == "__main__":
    main()