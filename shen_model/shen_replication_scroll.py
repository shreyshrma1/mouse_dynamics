"""
shen_replication_scroll.py

Identical to shen_replication.py, but uses extract_features_scroll to
add 9 scroll features to each window's holistic feature vector.

Adds StandardScaler normalization: fit on training windows, applied to
all windows before the Shen distance preprocessing.

Usage:
    python measurements/shen_replication_scroll.py
    python measurements/shen_replication_scroll.py --nu 0.06 --gamma scale
"""

import sys
import os
import argparse
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_scroll import extract_session_features

BALABIT_USERS = [
    "user7", "user9", "user12", "user15", "user16",
    "user20", "user21", "user23", "user29", "user35",
]

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

FEATURE_COLS = HOLISTIC_COLS + SCROLL_COLS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", type=float, default=0.06)
    parser.add_argument("--gamma", default="scale",
                        help="OCSVM gamma: float or 'scale' (default: scale)")
    parser.add_argument("--window_size", type=int, default=50,
                        help="Actions per window (default: 50)")
    parser.add_argument("--held_out_frac", type=float, default=0.25,
                        help="Fraction of windows to hold out for testing (default: 0.25)")
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


def score_windows(vecs, scaler, reference, dist_mean, dist_std, model):
    if len(vecs) == 0:
        return np.array([])
    x = scaler.transform(np.array(vecs))
    d = distance_vectors(x, reference)
    n = normalize(d, dist_mean, dist_std)
    return model.decision_function(n)


def main():
    args = parse_args()

    gamma = args.gamma
    try:
        gamma = float(gamma)
    except ValueError:
        pass

    print(f"nu={args.nu}, gamma={gamma}, window_size={args.window_size}, "
          f"held_out_frac={args.held_out_frac}")
    print(f"Feature vector: {len(HOLISTIC_COLS)} holistic + {len(SCROLL_COLS)} scroll "
          f"= {len(FEATURE_COLS)} total")
    print(f"Standardization: ON (fit on training windows per user)\n")

    all_results = []

    for user in BALABIT_USERS:
        print(f"{'=' * 50}")
        print(f"User: {user}")
        print(f"{'=' * 50}")

        user_dir = os.path.join(DATA_DIR, user)
        all_files = get_session_files(user_dir)
        all_vecs = extract_all_windows(all_files, user, args.window_size)

        if len(all_vecs) < 8:
            print(f"  Not enough windows ({len(all_vecs)}), skipping")
            continue

        n_total = len(all_vecs)
        n_test = max(1, int(n_total * args.held_out_frac))
        n_train = n_total - n_test

        train_samples = np.array(all_vecs[:n_train])
        test_samples = np.array(all_vecs[n_train:])

        print(f"  Total windows: {n_total}  |  Train: {n_train}  |  Test: {n_test}")

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_samples)

        reference = find_reference(train_scaled)
        train_dists = distance_vectors(train_scaled, reference)
        dist_mean = train_dists.mean(axis=0)
        dist_std = train_dists.std(axis=0)
        train_norm = normalize(train_dists, dist_mean, dist_std)

        model = OneClassSVM(kernel="rbf", nu=args.nu, gamma=gamma)
        model.fit(train_norm)

        train_scores = model.decision_function(train_norm)
        print(f"  Train scores: min={train_scores.min():.4f}, "
              f"mean={train_scores.mean():.4f}, max={train_scores.max():.4f}")

        legit_scores = score_windows(test_samples, scaler, reference,
                                     dist_mean, dist_std, model)
        legit_accepted = int(np.sum(legit_scores >= 0))
        legit_scored = len(legit_scores)

        print(f"\n  Legitimate held-out ({legit_scored} windows):")
        print(f"    Scores: min={legit_scores.min():+.4f}, "
              f"mean={legit_scores.mean():+.4f}, max={legit_scores.max():+.4f}")
        print(f"    Accepted: {legit_accepted}/{legit_scored}")

        impostor_accepted = 0
        impostor_scored = 0
        all_impostor_scores = []

        for other_user in BALABIT_USERS:
            if other_user == user:
                continue
            other_files = get_session_files(os.path.join(DATA_DIR, other_user))
            other_vecs = extract_all_windows(other_files, other_user, args.window_size)
            imp_scores = score_windows(other_vecs, scaler, reference,
                                       dist_mean, dist_std, model)
            if len(imp_scores) == 0:
                continue
            impostor_scored += len(imp_scores)
            impostor_accepted += int(np.sum(imp_scores >= 0))
            all_impostor_scores.extend(imp_scores.tolist())

        impostor_rejected = impostor_scored - impostor_accepted

        all_scores = np.concatenate([legit_scores, np.array(all_impostor_scores)])
        all_labels = np.concatenate([
            np.ones(len(legit_scores)),
            np.zeros(len(all_impostor_scores))
        ])
        auc = roc_auc_score(all_labels, all_scores) \
              if len(np.unique(all_labels)) == 2 else float("nan")

        frr = 1 - legit_accepted / legit_scored if legit_scored > 0 else 0.0
        far = impostor_accepted / impostor_scored if impostor_scored > 0 else 0.0
        acc = (legit_accepted + impostor_rejected) / (legit_scored + impostor_scored) \
              if (legit_scored + impostor_scored) > 0 else 0.0

        print(f"\n  Impostors:  {impostor_rejected}/{impostor_scored} rejected")
        print(f"  FAR: {far*100:.1f}%  FRR: {frr*100:.1f}%  "
              f"Accuracy: {acc*100:.1f}%  AUC: {auc:.4f}\n")

        all_results.append({
            "user": user,
            "legit_accepted": legit_accepted, "legit_scored": legit_scored,
            "impostor_rejected": impostor_rejected, "impostor_scored": impostor_scored,
            "far": far, "frr": frr, "acc": acc, "auc": auc,
            "all_scores": all_scores, "all_labels": all_labels,
        })

    if not all_results:
        print("No results.")
        return

    mean_far = sum(r["far"] for r in all_results) / len(all_results)
    mean_frr = sum(r["frr"] for r in all_results) / len(all_results)
    mean_acc = sum(r["acc"] for r in all_results) / len(all_results)
    mean_auc = sum(r["auc"] for r in all_results if not np.isnan(r["auc"])) \
               / sum(1 for r in all_results if not np.isnan(r["auc"]))

    tla = sum(r["legit_accepted"] for r in all_results)
    tls = sum(r["legit_scored"] for r in all_results)
    tir = sum(r["impostor_rejected"] for r in all_results)
    tis = sum(r["impostor_scored"] for r in all_results)

    micro_all_scores = np.concatenate([r["all_scores"] for r in all_results])
    micro_all_labels = np.concatenate([r["all_labels"] for r in all_results])
    micro_auc = roc_auc_score(micro_all_labels, micro_all_scores) \
                if len(np.unique(micro_all_labels)) == 2 else float("nan")

    micro_frr = 1 - tla / tls if tls > 0 else 0.0
    micro_far = (tis - tir) / tis if tis > 0 else 0.0
    micro_acc = (tla + tir) / (tls + tis) if (tls + tis) > 0 else 0.0

    print(f"\n{'=' * 62}")
    print(f"  Aggregate Results ({len(all_results)} users)")
    print(f"{'=' * 62}")
    print(f"  {'User':<12} {'FAR':>8} {'FRR':>8} {'Accuracy':>10} {'AUC':>8}")
    print(f"  {'-' * 50}")
    for r in all_results:
        print(f"  {r['user']:<12} {r['far']*100:>7.1f}% "
              f"{r['frr']*100:>7.1f}% {r['acc']*100:>9.1f}% {r['auc']:>8.4f}")
    print(f"  {'-' * 50}")
    print(f"  {'Mean':<12} {mean_far*100:>7.1f}% "
          f"{mean_frr*100:>7.1f}% {mean_acc*100:>9.1f}% {mean_auc:>8.4f}")
    print(f"  {'Micro':<12} {micro_far*100:>7.1f}% "
          f"{micro_frr*100:>7.1f}% {micro_acc*100:>9.1f}% {micro_auc:>8.4f}")
    print(f"\n  Total legitimate: {tla}/{tls} accepted")
    print(f"  Total impostors:  {tir}/{tis} rejected")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()