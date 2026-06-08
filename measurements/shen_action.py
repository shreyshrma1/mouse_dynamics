"""
shen_action.py

Extends the Shen et al. (2013) pipeline with procedural features.

Compact distance vector (48-dim total):
  - Holistic/action-composition (42-dim): Manhattan distance between
    [39-dim global averaged holistic vector, prop_MM, prop_PC, prop_DD]
  - Procedural (6-dim): for each action type (MM, PC, DD) and each curve
    type (speed, accel), compute the mean DTW distance between all curves
    of that type in the new window and all curves of that type in the
    reference window.

This version intentionally DOES NOT use the 117-dim per-action-type holistic
split. It keeps the older global 39-dim holistic average and adds only three
extra action-composition features.

Usage:
    python measurements/shen_action.py
    python measurements/shen_action.py --nu 0.06 --window_size 50
    python measurements/shen_action.py --max_impostor_windows 30
"""

import sys
import os
import argparse
import numpy as np
from numba import njit
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_action import (
    extract_procedural_windows,
    HOLISTIC_DIM,
)
from measurements.extract_features_sess import ACTION_MM, ACTION_PC, ACTION_DD

BALABIT_USERS = [
    "user7", "user9", "user12", "user15", "user16",
    "user20", "user21", "user23", "user29", "user35",
]

ACTION_TYPES = [ACTION_MM, ACTION_PC, ACTION_DD]
ACTION_NAMES = {ACTION_MM: "MM", ACTION_PC: "PC", ACTION_DD: "DD"}

DATA_DIR = "balabit_dataset/training_files"
HELD_OUT_FRAC = 0.25
PROCEDURAL_DIM = len(ACTION_TYPES) * 2  # speed + accel per action type
DISTANCE_DIM = HOLISTIC_DIM + PROCEDURAL_DIM  # 42 + 6 = 48


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", type=float, default=0.06)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--held_out_frac", type=float, default=0.25)
    parser.add_argument("--max_impostor_windows", type=int, default=30)
    return parser.parse_args()


def get_session_files(user_dir):
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


@njit
def dtw_distance(a, b):
    n, m = len(a), len(b)
    cost = np.full((n, m), np.inf)
    cost[0, 0] = abs(a[0] - b[0])

    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + abs(a[i] - b[0])

    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + abs(a[0] - b[j])

    for i in range(1, n):
        for j in range(1, m):
            cost[i, j] = abs(a[i] - b[j]) + min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )

    return cost[-1, -1]


def mean_dtw(new_curves, ref_curves, curve_idx):
    """
    Mean DTW distance between all curves of a given index (0=speed, 1=accel)
    in new_curves vs all curves of the same index in ref_curves.
    Returns 0.0 if either list is empty.
    """
    if not new_curves or not ref_curves:
        return 0.0

    dists = []
    for nc in new_curves:
        for rc in ref_curves:
            dists.append(dtw_distance(nc[curve_idx], rc[curve_idx]))

    return float(np.mean(dists))


def compute_distance_vector(window, ref_window):
    """
    42-dim holistic/action-composition Manhattan distance +
    6-dim procedural distance = 48 dimensions total.
    """
    holistic_dist = np.abs(window["holistic_vec"] - ref_window["holistic_vec"])

    procedural_dists = []
    for atype in ACTION_TYPES:
        new_curves = window["curves_by_type"].get(atype, [])
        ref_curves = ref_window["curves_by_type"].get(atype, [])
        procedural_dists.append(mean_dtw(new_curves, ref_curves, 0))  # speed
        procedural_dists.append(mean_dtw(new_curves, ref_curves, 1))  # accel

    dist_vec = np.concatenate([holistic_dist, np.array(procedural_dists)])

    if dist_vec.shape[0] != DISTANCE_DIM:
        raise ValueError(
            f"Expected distance dim {DISTANCE_DIM}, got {dist_vec.shape[0]}"
        )

    return dist_vec


def compute_all_distance_vectors(windows, ref_window):
    return np.array([compute_distance_vector(w, ref_window) for w in windows])


def find_reference_idx(holistic_vecs):
    n = len(holistic_vecs)
    mean_dists = np.zeros(n)

    for i in range(n):
        dists = np.sum(np.abs(holistic_vecs - holistic_vecs[i]), axis=1)
        # Exclude self-distance when possible.
        mean_dists[i] = dists.sum() / max(n - 1, 1)

    return int(np.argmin(mean_dists))


def normalize(dist_vecs, mean, std):
    std_safe = np.where(std < 1e-9, 1.0, std)
    return (dist_vecs - mean) / std_safe


def sample_windows(windows, max_count):
    if len(windows) <= max_count:
        return windows

    indices = np.linspace(0, len(windows) - 1, max_count, dtype=int)
    return [windows[i] for i in indices]


def main():
    args = parse_args()

    gamma = args.gamma
    try:
        gamma = float(gamma)
    except ValueError:
        pass

    print(f"nu={args.nu}, gamma={gamma}, window_size={args.window_size}, "
          f"held_out_frac={args.held_out_frac}, "
          f"max_impostor_windows={args.max_impostor_windows}")
    print(f"holistic_dim={HOLISTIC_DIM}, procedural_dim={PROCEDURAL_DIM}, "
          f"distance_dim={DISTANCE_DIM}\n")

    # Warm up JIT.
    print("Warming up JIT compiler...")
    _a = np.array([1.0, 2.0, 3.0])
    dtw_distance(_a, _a)
    print("JIT ready.\n")

    # Precompute all windows once.
    print("Precomputing windows for all users...")
    all_user_windows = {}

    for user in BALABIT_USERS:
        user_dir = os.path.join(DATA_DIR, user)
        all_files = get_session_files(user_dir)
        wins = extract_procedural_windows(all_files, user, args.window_size)
        all_user_windows[user] = wins
        print(f"  {user:<12} {len(wins)} windows")

    print()

    all_results = []

    for user in BALABIT_USERS:
        print(f"{'=' * 50}")
        print(f"User: {user}")
        print(f"{'=' * 50}")

        all_wins = all_user_windows[user]

        if len(all_wins) < 8:
            print(f"  Not enough windows ({len(all_wins)}), skipping")
            continue

        n_total = len(all_wins)
        n_test = max(1, int(n_total * args.held_out_frac))
        n_train = n_total - n_test

        train_wins = all_wins[:n_train]
        test_wins = all_wins[n_train:]

        print(f"  Total windows: {n_total}  |  Train: {n_train}  |  Test: {n_test}")

        holistic_vecs = np.array([w["holistic_vec"] for w in train_wins])
        ref_idx = find_reference_idx(holistic_vecs)
        ref_window = train_wins[ref_idx]

        ref_counts = {
            ACTION_NAMES[t]: len(ref_window["curves_by_type"].get(t, []))
            for t in ACTION_TYPES
        }

        print(f"  Reference: window {ref_idx}, "
              f"MM={ref_counts['MM']} PC={ref_counts['PC']} "
              f"DD={ref_counts['DD']} actions, "
              f"distance vector dim: {DISTANCE_DIM}")

        print("  Computing training distance vectors...")
        train_dists = compute_all_distance_vectors(train_wins, ref_window)
        dist_mean = train_dists.mean(axis=0)
        dist_std = train_dists.std(axis=0)
        train_norm = normalize(train_dists, dist_mean, dist_std)

        model = OneClassSVM(kernel="rbf", nu=args.nu, gamma=gamma)
        model.fit(train_norm)

        train_scores = model.decision_function(train_norm)
        print(f"  Train scores: min={train_scores.min():.4f}, "
              f"mean={train_scores.mean():.4f}, max={train_scores.max():.4f}")

        print("  Computing test distance vectors...")
        test_dists = compute_all_distance_vectors(test_wins, ref_window)
        test_norm = normalize(test_dists, dist_mean, dist_std)
        legit_scores = model.decision_function(test_norm)

        legit_accepted = int(np.sum(legit_scores >= 0))
        legit_scored = len(legit_scores)

        print(f"\n  Legitimate held-out ({legit_scored} windows):")
        print(f"    Scores: min={legit_scores.min():+.4f}, "
              f"mean={legit_scores.mean():+.4f}, max={legit_scores.max():+.4f}")
        print(f"    Accepted: {legit_accepted}/{legit_scored}")

        impostor_accepted = 0
        impostor_scored = 0
        all_impostor_scores = []

        print(f"\n  Scoring impostors (max {args.max_impostor_windows} windows/user)...")

        for other_user in BALABIT_USERS:
            if other_user == user:
                continue

            other_wins = sample_windows(
                all_user_windows[other_user],
                args.max_impostor_windows,
            )

            if not other_wins:
                continue

            other_dists = compute_all_distance_vectors(other_wins, ref_window)
            other_norm = normalize(other_dists, dist_mean, dist_std)
            imp_scores = model.decision_function(other_norm)

            impostor_scored += len(imp_scores)
            impostor_accepted += int(np.sum(imp_scores >= 0))
            all_impostor_scores.extend(imp_scores.tolist())

            print(f"    {other_user:<12} "
                  f"{int(np.sum(imp_scores < 0))}/{len(imp_scores)} rejected")

        impostor_rejected = impostor_scored - impostor_accepted

        all_scores = np.concatenate([legit_scores, np.array(all_impostor_scores)])
        all_labels = np.concatenate([
            np.ones(len(legit_scores)),
            np.zeros(len(all_impostor_scores)),
        ])

        auc = (
            roc_auc_score(all_labels, all_scores)
            if len(np.unique(all_labels)) == 2
            else float("nan")
        )

        frr = 1 - legit_accepted / legit_scored if legit_scored > 0 else 0.0
        far = impostor_accepted / impostor_scored if impostor_scored > 0 else 0.0
        acc = (
            (legit_accepted + impostor_rejected) /
            (legit_scored + impostor_scored)
            if (legit_scored + impostor_scored) > 0
            else 0.0
        )

        print(f"\n  Impostors:  {impostor_rejected}/{impostor_scored} rejected")
        print(f"  FAR: {far * 100:.1f}%  FRR: {frr * 100:.1f}%  "
              f"Accuracy: {acc * 100:.1f}%  AUC: {auc:.4f}\n")

        all_results.append({
            "user": user,
            "legit_accepted": legit_accepted,
            "legit_scored": legit_scored,
            "impostor_rejected": impostor_rejected,
            "impostor_scored": impostor_scored,
            "far": far,
            "frr": frr,
            "acc": acc,
            "auc": auc,
            "all_scores": all_scores,
            "all_labels": all_labels,
        })

    if not all_results:
        print("No results.")
        return

    mean_far = sum(r["far"] for r in all_results) / len(all_results)
    mean_frr = sum(r["frr"] for r in all_results) / len(all_results)
    mean_acc = sum(r["acc"] for r in all_results) / len(all_results)

    valid_aucs = [r["auc"] for r in all_results if not np.isnan(r["auc"])]
    mean_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else float("nan")

    tla = sum(r["legit_accepted"] for r in all_results)
    tls = sum(r["legit_scored"] for r in all_results)
    tir = sum(r["impostor_rejected"] for r in all_results)
    tis = sum(r["impostor_scored"] for r in all_results)

    micro_all_scores = np.concatenate([r["all_scores"] for r in all_results])
    micro_all_labels = np.concatenate([r["all_labels"] for r in all_results])

    micro_auc = (
        roc_auc_score(micro_all_labels, micro_all_scores)
        if len(np.unique(micro_all_labels)) == 2
        else float("nan")
    )

    micro_frr = 1 - tla / tls if tls > 0 else 0.0
    micro_far = (tis - tir) / tis if tis > 0 else 0.0
    micro_acc = (tla + tir) / (tls + tis) if (tls + tis) > 0 else 0.0

    print(f"\n{'=' * 62}")
    print(f"  Aggregate Results ({len(all_results)} users)")
    print(f"{'=' * 62}")
    print(f"  {'User':<12} {'FAR':>8} {'FRR':>8} {'Accuracy':>10} {'AUC':>8}")
    print(f"  {'-' * 50}")

    for r in all_results:
        print(f"  {r['user']:<12} {r['far'] * 100:>7.1f}% "
              f"{r['frr'] * 100:>7.1f}% "
              f"{r['acc'] * 100:>9.1f}% "
              f"{r['auc']:>8.4f}")

    print(f"  {'-' * 50}")
    print(f"  {'Mean':<12} {mean_far * 100:>7.1f}% "
          f"{mean_frr * 100:>7.1f}% "
          f"{mean_acc * 100:>9.1f}% "
          f"{mean_auc:>8.4f}")

    print(f"  {'Micro':<12} {micro_far * 100:>7.1f}% "
          f"{micro_frr * 100:>7.1f}% "
          f"{micro_acc * 100:>9.1f}% "
          f"{micro_auc:>8.4f}")

    print(f"\n  Total legitimate: {tla}/{tls} accepted")
    print(f"  Total impostors:  {tir}/{tis} rejected")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()