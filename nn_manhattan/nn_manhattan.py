"""
nn_manhattan.py

Identical to shen_limited.py but replaces the OneClassSVM with the
Nearest Neighbor (Manhattan) one-class detector described in:

    Shen, Cai, Guan & Maxion (2014). Performance evaluation of
    anomaly-detection algorithms for mouse dynamics.
    Computers & Security, 45, 156-171.

The detector works as follows:
  - Training: store all preprocessed legitimate user feature vectors.
  - Scoring:  for each test vector, compute the average Manhattan distance
              to its k nearest neighbours in the training set.
              Lower distance = closer to legitimate behaviour.
              Negated so that higher score = more legitimate (ACCEPT).

The EER threshold (where FAR ≈ FRR) is used for accept/reject decisions,
matching the evaluation methodology in the paper.

Usage:
    python measurements/nn_manhattan.py
    python measurements/nn_manhattan.py --k 3 --top_n 15
"""

import sys
import os
import argparse
import numpy as np
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
    "scroll_count", "scroll_rate", "scroll_ratio", "scroll_up_ratio",
    "scroll_dur_mean", "scroll_dur_std",
    "scroll_burst_count", "scroll_burst_dur_mean", "scroll_burst_len_mean",
]

ALL_FEATURE_COLS = HOLISTIC_COLS + SCROLL_COLS

TOP_FEATURES = [
    "num_critical_points", "num_points", "sum_of_angles", "scroll_rate",
    "scroll_burst_len_mean", "sd_omega", "scroll_count", "straightness",
    "max_vx", "sd_curv", "scroll_ratio", "max_a", "mean_omega", "min_vx", "min_vy",
]

TOP_INDICES = None  # set at runtime based on --top_n


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",             type=int,   default=3,
                        help="Number of nearest neighbours (default: 3, as per Shen et al.)")
    parser.add_argument("--window_size",   type=int,   default=50)
    parser.add_argument("--held_out_frac", type=float, default=0.25)
    parser.add_argument("--top_n",         type=int,   default=15,
                        help="Number of top features to use, 1-15 (default: 15)")
    return parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

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
            df = df.replace([np.inf, -np.inf], float("nan")).dropna(subset=ALL_FEATURE_COLS)
            if len(df) == 0:
                continue
            for _, grp in df.groupby("session"):
                rows = grp[ALL_FEATURE_COLS].values
                if len(rows) >= 1:
                    all_vecs.append(rows.mean(axis=0)[TOP_INDICES])
        except Exception as e:
            print(f"  [!] {os.path.basename(path)}: {e}")
    return all_vecs


# ── Shen preprocessing ────────────────────────────────────────────────────────

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


def preprocess(vecs, scaler, reference, dist_mean, dist_std):
    x = scaler.transform(np.array(vecs))
    d = distance_vectors(x, reference)
    return normalize(d, dist_mean, dist_std)


# ── NN Manhattan detector ─────────────────────────────────────────────────────

def nn_manhattan_score(test_vecs, train_vecs, k):
    """
    For each test vector, compute the average Manhattan distance to its
    k nearest neighbours in train_vecs.
    Negated so higher score = closer to legitimate = more likely to ACCEPT.
    """
    test_vecs  = np.atleast_2d(test_vecs)
    scores     = np.zeros(len(test_vecs))
    for i, test_vec in enumerate(test_vecs):
        dists      = np.sum(np.abs(train_vecs - test_vec), axis=1)
        knn_dists  = np.sort(dists)[:k]
        scores[i]  = knn_dists.mean()
    return -scores


def find_eer_threshold(legit_scores, impostor_scores):
    """Find the threshold at which FAR ≈ FRR."""
    all_scores  = np.concatenate([legit_scores, impostor_scores])
    thresholds  = np.unique(all_scores)
    best_thresh = thresholds[0]
    best_diff   = float("inf")
    for t in thresholds:
        frr  = np.mean(legit_scores    < t)
        far  = np.mean(impostor_scores >= t)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff   = diff
            best_thresh = t
    eer = (np.mean(legit_scores < best_thresh) +
           np.mean(impostor_scores >= best_thresh)) / 2
    return best_thresh, eer


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    top_n = max(1, min(args.top_n, len(TOP_FEATURES)))
    global TOP_INDICES
    TOP_INDICES = [ALL_FEATURE_COLS.index(f) for f in TOP_FEATURES[:top_n]]

    print(f"k={args.k}, window_size={args.window_size}, "
          f"held_out_frac={args.held_out_frac}")
    print(f"Feature vector: {top_n} features (top-{top_n} subset)")
    print(f"Standardization: ON (fit on training windows per user)\n")

    all_results = []

    for user in BALABIT_USERS:
        print(f"{'=' * 50}")
        print(f"User: {user}")
        print(f"{'=' * 50}")

        user_dir  = os.path.join(DATA_DIR, user)
        all_files = get_session_files(user_dir)
        all_vecs  = extract_all_windows(all_files, user, args.window_size)

        if len(all_vecs) < 8:
            print(f"  Not enough windows ({len(all_vecs)}), skipping")
            continue

        n_total = len(all_vecs)
        n_test  = max(1, int(n_total * args.held_out_frac))
        n_train = n_total - n_test

        train_samples = np.array(all_vecs[:n_train])
        test_samples  = np.array(all_vecs[n_train:])

        print(f"  Total windows: {n_total}  |  Train: {n_train}  |  Test: {n_test}")

        scaler       = StandardScaler()
        train_scaled = scaler.fit_transform(train_samples)
        reference    = find_reference(train_scaled)
        train_dists  = distance_vectors(train_scaled, reference)
        dist_mean    = train_dists.mean(axis=0)
        dist_std     = train_dists.std(axis=0)
        train_norm   = normalize(train_dists, dist_mean, dist_std)

        # Legitimate held-out scores
        test_norm    = preprocess(test_samples, scaler, reference, dist_mean, dist_std)
        legit_scores = nn_manhattan_score(test_norm, train_norm, k=args.k)

        print(f"\n  Legitimate held-out ({n_test} windows):")
        print(f"    Scores: min={legit_scores.min():+.4f}, "
              f"mean={legit_scores.mean():+.4f}, max={legit_scores.max():+.4f}")

        # Impostor scores
        impostor_scored     = 0
        all_impostor_scores = []

        for other_user in BALABIT_USERS:
            if other_user == user:
                continue
            other_files = get_session_files(os.path.join(DATA_DIR, other_user))
            other_vecs  = extract_all_windows(other_files, other_user, args.window_size)
            if not other_vecs:
                continue
            other_norm  = preprocess(other_vecs, scaler, reference, dist_mean, dist_std)
            imp_scores  = nn_manhattan_score(other_norm, train_norm, k=args.k)
            impostor_scored += len(imp_scores)
            all_impostor_scores.extend(imp_scores.tolist())

        all_impostor_scores = np.array(all_impostor_scores)

        # EER threshold
        threshold, eer = find_eer_threshold(legit_scores, all_impostor_scores)

        legit_accepted    = int(np.sum(legit_scores        >= threshold))
        print(f"    Accepted: {legit_accepted}/{n_test}")
        impostor_accepted = int(np.sum(all_impostor_scores >= threshold))
        impostor_rejected = impostor_scored - impostor_accepted

        all_scores = np.concatenate([legit_scores, all_impostor_scores])
        all_labels = np.concatenate([
            np.ones(len(legit_scores)),
            np.zeros(len(all_impostor_scores))
        ])
        auc = roc_auc_score(all_labels, all_scores) \
              if len(np.unique(all_labels)) == 2 else float("nan")

        frr = 1 - legit_accepted    / n_test          if n_test          > 0 else 0.0
        far = impostor_accepted      / impostor_scored if impostor_scored > 0 else 0.0
        acc = (legit_accepted + impostor_rejected) / (n_test + impostor_scored) \
              if (n_test + impostor_scored) > 0 else 0.0

        print(f"\n  Impostors:  {impostor_rejected}/{impostor_scored} rejected")
        print(f"  FAR: {far*100:.1f}%  FRR: {frr*100:.1f}%  "
              f"Accuracy: {acc*100:.1f}%  AUC: {auc:.4f}\n")

        all_results.append({
            "user":              user,
            "legit_accepted":    legit_accepted,
            "legit_scored":      n_test,
            "impostor_rejected": impostor_rejected,
            "impostor_scored":   impostor_scored,
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

    tla = sum(r["legit_accepted"]    for r in all_results)
    tls = sum(r["legit_scored"]      for r in all_results)
    tir = sum(r["impostor_rejected"] for r in all_results)
    tis = sum(r["impostor_scored"]   for r in all_results)

    micro_all_scores = np.concatenate([r["all_scores"] for r in all_results])
    micro_all_labels = np.concatenate([r["all_labels"] for r in all_results])
    micro_auc = roc_auc_score(micro_all_labels, micro_all_scores) \
                if len(np.unique(micro_all_labels)) == 2 else float("nan")

    micro_frr = 1 - tla / tls            if tls > 0 else 0.0
    micro_far = (tis - tir) / tis        if tis > 0 else 0.0
    micro_acc = (tla + tir) / (tls + tis) if (tls + tis) > 0 else 0.0

    print(f"\n{'=' * 62}")
    print(f"  Aggregate Results ({len(all_results)} users) — top-{top_n} features, k={args.k}")
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