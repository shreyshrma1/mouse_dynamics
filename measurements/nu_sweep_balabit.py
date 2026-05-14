"""
nu_sweep_balabit.py

Sweeps nu values on the Balabit dataset. For each nu, trains and evaluates
the Shen pipeline on all 10 users and reports mean FAR, FRR, and AUC.

Usage:
    python measurements/nu_sweep_balabit.py
    python measurements/nu_sweep_balabit.py --window_size 30
"""

import sys
import os
import argparse
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_sess import extract_session_features

BALABIT_USERS = [
    "user7", "user9", "user12", "user15", "user16",
    "user20", "user21", "user23", "user29", "user35",
]

DATA_DIR      = "balabit_dataset/training_files"
GAMMA         = "scale"
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=30)
    return parser.parse_args()


def get_session_files(user_dir):
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def extract_all_windows(session_files, user_id, window_size):
    all_vecs = []
    for path in session_files:
        try:
            df = extract_session_features(path, user_id, window_size=window_size)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
            if len(df) == 0:
                continue
            for _, grp in df.groupby("session"):
                rows = grp[FEATURE_COLS].values
                if len(rows) >= 1:
                    all_vecs.append(rows.mean(axis=0))
        except Exception:
            pass
    return all_vecs


def find_reference(train_samples):
    n = len(train_samples)
    mean_dists = np.zeros(n)
    for i in range(n):
        dists = np.sum(np.abs(train_samples - train_samples[i]), axis=1)
        mean_dists[i] = dists.sum() / max(n - 1, 1)
    return train_samples[np.argmin(mean_dists)]


def main():
    args = parse_args()
    window_size = args.window_size

    print(f"window_size={window_size}, gamma={GAMMA}, held_out_frac={HELD_OUT_FRAC}\n")

    # Precompute all windows for all users once
    print("Extracting windows for all users...")
    user_data = {}
    for user in BALABIT_USERS:
        all_files = get_session_files(os.path.join(DATA_DIR, user))
        all_vecs  = extract_all_windows(all_files, user, window_size)
        user_data[user] = all_vecs
        print(f"  {user:<12} {len(all_vecs)} windows")
    print()

    summary = []

    for nu in NU_GRID:
        print(f"{'=' * 58}")
        print(f"nu = {nu}")
        print(f"{'=' * 58}")
        print(f"  {'User':<12} {'n_train':>8} {'FAR':>8} {'FRR':>8} {'AUC':>8}")
        print(f"  {'-' * 48}")

        user_aucs = []
        user_fars = []
        user_frrs = []

        for user in BALABIT_USERS:
            all_vecs = user_data[user]
            if len(all_vecs) < 8:
                print(f"  {user:<12} skipped")
                continue

            n_total = len(all_vecs)
            n_test  = max(1, int(n_total * HELD_OUT_FRAC))
            n_train = n_total - n_test

            train_samples = np.array(all_vecs[:n_train])
            test_samples  = np.array(all_vecs[n_train:])

            reference   = find_reference(train_samples)
            train_dists = np.abs(train_samples - reference)
            dist_mean   = train_dists.mean(axis=0)
            dist_std    = train_dists.std(axis=0)
            std_safe    = np.where(dist_std < 1e-9, 1.0, dist_std)
            train_norm  = (train_dists - dist_mean) / std_safe

            model = OneClassSVM(kernel="rbf", nu=nu, gamma=GAMMA)
            model.fit(train_norm)

            def score(vecs):
                if len(vecs) == 0:
                    return np.array([])
                x = np.abs(np.array(vecs) - reference)
                x = (x - dist_mean) / std_safe
                return model.decision_function(x)

            legit_scores = score(test_samples)

            imp_scores_all = []
            for other_user in BALABIT_USERS:
                if other_user == user:
                    continue
                s = score(user_data[other_user])
                if len(s) > 0:
                    imp_scores_all.extend(s.tolist())

            if not imp_scores_all:
                continue

            imp_scores = np.array(imp_scores_all)
            frr = 1 - float(np.mean(legit_scores >= 0))
            far = float(np.mean(imp_scores >= 0))

            all_scores = np.concatenate([legit_scores, imp_scores])
            all_labels = np.concatenate([np.ones(len(legit_scores)),
                                         np.zeros(len(imp_scores))])
            auc = roc_auc_score(all_labels, all_scores) \
                  if len(np.unique(all_labels)) == 2 else float("nan")

            user_aucs.append(auc)
            user_fars.append(far)
            user_frrs.append(frr)

            print(f"  {user:<12} {n_train:>8} {far*100:>7.1f}% "
                  f"{frr*100:>7.1f}% {auc:>8.4f}")

        if user_aucs:
            mean_auc = float(np.mean(user_aucs))
            mean_far = float(np.mean(user_fars))
            mean_frr = float(np.mean(user_frrs))
            print(f"  {'-' * 48}")
            print(f"  {'Mean':<12}          {mean_far*100:>7.1f}% "
                  f"{mean_frr*100:>7.1f}% {mean_auc:>8.4f}\n")
            summary.append((nu, mean_far, mean_frr, mean_auc))

    print(f"\n{'=' * 55}")
    print(f"  Summary (window_size={window_size})")
    print(f"{'=' * 55}")
    print(f"  {'nu':<8} {'Mean FAR':>10} {'Mean FRR':>10} {'Mean AUC':>10}")
    print(f"  {'-' * 42}")
    for nu, mean_far, mean_frr, mean_auc in summary:
        print(f"  {nu:<8} {mean_far*100:>9.1f}% {mean_frr*100:>9.1f}% {mean_auc:>10.4f}")

    if summary:
        best = max(summary, key=lambda x: x[3])
        print(f"\n  Best nu by AUC: {best[0]} "
              f"(AUC={best[3]:.4f}, FAR={best[1]*100:.1f}%, FRR={best[2]*100:.1f}%)")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()