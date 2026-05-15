"""
nu_sweep.py

Sweeps nu values on a user's collected data to find the optimal nu.
Shows per-window scores for legitimate held-out windows at each nu value.

Usage:
    python measurements/nu_sweep.py
"""

import sys
import os
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_sess import extract_session_features

DATA_DIR      = "collected_data"
IMPOSTOR_DIR  = "balabit_dataset/training_files"
GAMMA         = "scale"
WINDOW_SIZE   = 10
HELD_OUT_FRAC = 0.25

NU_GRID = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

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


def get_session_files(directory):
    if not os.path.isdir(directory):
        return []
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
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


def main():
    user_id = input("Enter user ID: ").strip()

    session_files = get_session_files(os.path.join(DATA_DIR, user_id))
    all_vecs = extract_all_windows(session_files, user_id)

    if len(all_vecs) < 8:
        print(f"Not enough windows ({len(all_vecs)}) — collect more data")
        sys.exit(1)

    n_total = len(all_vecs)
    n_test  = max(1, int(n_total * HELD_OUT_FRAC))
    n_train = n_total - n_test

    train_samples = np.array(all_vecs[:n_train])
    test_samples  = np.array(all_vecs[n_train:])

    print(f"Total windows: {n_total}  |  Train: {n_train}  |  Test: {n_test}\n")

    # Precompute reference and normalization once
    reference   = find_reference(train_samples)
    train_dists = np.abs(train_samples - reference)
    dist_mean   = train_dists.mean(axis=0)
    dist_std    = train_dists.std(axis=0)
    std_safe    = np.where(dist_std < 1e-9, 1.0, dist_std)
    train_norm  = (train_dists - dist_mean) / std_safe

    # Precompute impostor windows once
    print("Extracting impostor windows...")
    imp_vecs_all = []
    for imp_user in sorted(os.listdir(IMPOSTOR_DIR)):
        imp_files = get_session_files(os.path.join(IMPOSTOR_DIR, imp_user))
        for path in imp_files:
            try:
                df = extract_session_features(path, imp_user, window_size=WINDOW_SIZE)
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
                if len(df) == 0:
                    continue
                for _, grp in df.groupby("session"):
                    rows = grp[FEATURE_COLS].values
                    if len(rows) >= 1:
                        imp_vecs_all.append(rows.mean(axis=0))
            except Exception:
                pass

    print(f"Impostor windows: {len(imp_vecs_all)}\n")

    def score(vecs):
        x = np.abs(np.array(vecs) - reference)
        x = (x - dist_mean) / std_safe
        return model.decision_function(x)

    summary = []

    for nu in NU_GRID:
        print(f"{'=' * 50}")
        print(f"nu = {nu}")
        print(f"{'=' * 50}")

        model = OneClassSVM(kernel="rbf", nu=nu, gamma=GAMMA)
        model.fit(train_norm)

        train_scores = model.decision_function(train_norm)
        train_acc    = float(np.mean(train_scores >= 0))
        print(f"  Train: {train_acc*100:.1f}% accepted  "
              f"(min={train_scores.min():.4f}, mean={train_scores.mean():.4f}, "
              f"max={train_scores.max():.4f})")

        # Per-window legitimate scores
        legit_scores = score(test_samples)
        print(f"\n  Legitimate held-out ({len(legit_scores)} windows):")
        for i, s in enumerate(legit_scores):
            marker = "✓" if s >= 0 else "✗"
            print(f"    window {i+1:>2}  {s:>+8.4f}  {marker}")
        legit_accepted = int(np.sum(legit_scores >= 0))
        frr = 1 - legit_accepted / len(legit_scores)
        print(f"  Accepted: {legit_accepted}/{len(legit_scores)}  FRR={frr*100:.1f}%")

        # Impostor summary
        imp_scores = score(imp_vecs_all)
        far = float(np.mean(imp_scores >= 0))
        print(f"\n  Impostors: {int(np.sum(imp_scores < 0))}/{len(imp_scores)} rejected  "
              f"FAR={far*100:.1f}%")

        all_scores = np.concatenate([legit_scores, imp_scores])
        all_labels = np.concatenate([np.ones(len(legit_scores)),
                                     np.zeros(len(imp_scores))])
        auc = roc_auc_score(all_labels, all_scores) \
              if len(np.unique(all_labels)) == 2 else float("nan")
        print(f"  AUC: {auc:.4f}\n")

        summary.append((nu, train_acc, frr, far, auc))

    print(f"\n{'=' * 50}")
    print(f"  Summary")
    print(f"{'=' * 50}")
    print(f"  {'nu':<8} {'Train acc':>10} {'FRR':>8} {'FAR':>8} {'AUC':>8}")
    print(f"  {'-' * 46}")
    for nu, train_acc, frr, far, auc in summary:
        print(f"  {nu:<8} {train_acc*100:>9.1f}% {frr*100:>7.1f}% "
              f"{far*100:>7.1f}% {auc:>8.4f}")

    best = max(summary, key=lambda x: x[4])
    print(f"\n  Best nu by AUC: {best[0]} "
          f"(AUC={best[4]:.4f}, FRR={best[2]*100:.1f}%, FAR={best[3]*100:.1f}%)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()