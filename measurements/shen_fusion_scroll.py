"""
shen_fusion_scroll.py

Implements the Kumar et al. (2017) one-class classifier fusion pipeline
on top of the Shen et al. feature extraction, with scroll features included.

Four classifiers are trained on each user's normalized distance vectors:
    - One-Class SVM (OCSVM)
    - Elliptic Envelope
    - Isolation Forest
    - Local Outlier Factor (novelty=True)

Their decision_function scores are averaged to produce a single fused score.
Everything else (windowing, reference vector, normalization, evaluation) is
identical to shen_replication_scroll.py.

Reference:
    Kumar, Kundu & Phoha (2017). "Continuous Authentication Using One-class
    Classifiers and their Fusion." https://arxiv.org/abs/1710.11075

Usage:
    python measurements/shen_fusion_scroll.py
    python measurements/shen_fusion_scroll.py --nu 0.06 --gamma scale
"""

import sys
import os
import argparse
import warnings
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    parser.add_argument("--nu",              type=float, default=0.06)
    parser.add_argument("--gamma",           default="scale",
                        help="OCSVM gamma: float or 'scale' (default: scale)")
    parser.add_argument("--contamination",   type=float, default=0.06,
                        help="Contamination for EE, IF, LOF (default: 0.06)")
    parser.add_argument("--window_size",     type=int,   default=50)
    parser.add_argument("--held_out_frac",   type=float, default=0.25)
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


def build_classifiers(nu, gamma, contamination):
    return {
        "ocsvm":    OneClassSVM(kernel="rbf", nu=nu, gamma=gamma),
        "elliptic": EllipticEnvelope(contamination=contamination, random_state=0),
        "iforest":  IsolationForest(contamination=contamination, random_state=0),
        "lof":      LocalOutlierFactor(novelty=True, contamination=contamination),
    }


def normalize_scores(scores, lo, hi):
    """Min-max normalize scores to [0, 1] using training score range."""
    rng = hi - lo
    if rng < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / rng


def find_eer_threshold(legit_scores, impostor_scores):
    """
    Find the threshold where FAR ≈ FRR (Equal Error Rate).
    Sweeps all unique score values and returns the threshold and EER.
    """
    all_scores = np.concatenate([legit_scores, impostor_scores])
    thresholds = np.unique(all_scores)

    best_thr = 0.5
    best_eer = 1.0
    best_far = 1.0
    best_frr = 1.0

    for thr in thresholds:
        frr = np.mean(legit_scores < thr)
        far = np.mean(impostor_scores >= thr)
        eer = abs(far - frr)
        if eer < best_eer:
            best_eer  = eer
            best_thr  = thr
            best_far  = far
            best_frr  = frr

    return best_thr, (best_far + best_frr) / 2.0


def fused_score(vecs, classifiers, score_ranges, reference, dist_mean, dist_std):
    """
    Score each classifier, normalize to [0,1] using training range,
    then average. Each classifier contributes equally regardless of
    its raw score scale.
    """
    if len(vecs) == 0:
        return np.array([])
    x = normalize(distance_vectors(np.array(vecs), reference), dist_mean, dist_std)
    normalized = []
    for name, clf in classifiers.items():
        lo, hi = score_ranges[name]
        raw = clf.decision_function(x)
        normalized.append(normalize_scores(raw, lo, hi))
    return np.stack(normalized, axis=0).mean(axis=0)


def main():
    args = parse_args()

    gamma = args.gamma
    try:
        gamma = float(gamma)
    except ValueError:
        pass

    print(f"nu={args.nu}, gamma={gamma}, contamination={args.contamination}, "
          f"window_size={args.window_size}, held_out_frac={args.held_out_frac}")
    print(f"Feature vector: {len(HOLISTIC_COLS)} holistic + {len(SCROLL_COLS)} scroll "
          f"= {len(FEATURE_COLS)} total")
    print(f"Classifiers: OCSVM + EllipticEnvelope + IsolationForest + LOF (fused by mean)\n")

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

        reference   = find_reference(train_samples)
        train_dists = distance_vectors(train_samples, reference)
        dist_mean   = train_dists.mean(axis=0)
        dist_std    = train_dists.std(axis=0)
        train_norm  = normalize(train_dists, dist_mean, dist_std)

        classifiers = build_classifiers(args.nu, gamma, args.contamination)
        for name, clf in classifiers.items():
            try:
                clf.fit(train_norm)
            except Exception as e:
                print(f"  [!] {name} failed to fit: {e}")

        # Compute per-classifier score ranges on training data for normalization
        score_ranges = {}
        raw_train_scores = {}
        for name, clf in classifiers.items():
            s = clf.decision_function(train_norm)
            score_ranges[name] = (s.min(), s.max())
            raw_train_scores[name] = s

        # Fused training score (normalized)
        train_fused = np.stack([
            normalize_scores(raw_train_scores[n], *score_ranges[n])
            for n in classifiers
        ], axis=0).mean(axis=0)
        print(f"  Train scores (fused, normalized): min={train_fused.min():.4f}, "
              f"mean={train_fused.mean():.4f}, max={train_fused.max():.4f}")
        for name in classifiers:
            lo, hi = score_ranges[name]
            print(f"    {name:<12} raw range: [{lo:.2f}, {hi:.2f}]")

        # Score all legitimate held-out windows
        legit_scores = fused_score(test_samples, classifiers, score_ranges, reference, dist_mean, dist_std)
        legit_scored = len(legit_scores)

        # Score all impostor windows
        all_impostor_scores = []
        for other_user in BALABIT_USERS:
            if other_user == user:
                continue
            other_files = get_session_files(os.path.join(DATA_DIR, other_user))
            other_vecs  = extract_all_windows(other_files, other_user, args.window_size)
            imp_scores  = fused_score(other_vecs, classifiers, score_ranges, reference, dist_mean, dist_std)
            if len(imp_scores) > 0:
                all_impostor_scores.extend(imp_scores.tolist())

        impostor_scores = np.array(all_impostor_scores)
        impostor_scored = len(impostor_scores)

        # Find EER threshold from held-out legit + all impostor scores
        eer_thr, eer = find_eer_threshold(legit_scores, impostor_scores)

        print(f"\n  Legitimate held-out ({legit_scored} windows):")
        print(f"    Scores: min={legit_scores.min():+.4f}, "
              f"mean={legit_scores.mean():+.4f}, max={legit_scores.max():+.4f}")
        print(f"  EER threshold: {eer_thr:.4f}  |  EER: {eer*100:.1f}%")

        # Evaluate at EER threshold
        legit_accepted    = int(np.sum(legit_scores   >= eer_thr))
        impostor_accepted = int(np.sum(impostor_scores >= eer_thr))
        impostor_rejected = impostor_scored - impostor_accepted

        all_scores = np.concatenate([legit_scores, impostor_scores])
        all_labels = np.concatenate([
            np.ones(legit_scored),
            np.zeros(impostor_scored)
        ])
        auc = roc_auc_score(all_labels, all_scores) \
              if len(np.unique(all_labels)) == 2 else float("nan")

        frr = 1 - legit_accepted  / legit_scored    if legit_scored    > 0 else 0.0
        far = impostor_accepted   / impostor_scored  if impostor_scored > 0 else 0.0
        acc = (legit_accepted + impostor_rejected) / (legit_scored + impostor_scored) \
              if (legit_scored + impostor_scored) > 0 else 0.0

        print(f"  Impostors:  {impostor_rejected}/{impostor_scored} rejected")
        print(f"  FAR: {far*100:.1f}%  FRR: {frr*100:.1f}%  "
              f"Accuracy: {acc*100:.1f}%  AUC: {auc:.4f}\n")

        all_results.append({
            "user": user,
            "legit_accepted": legit_accepted, "legit_scored": legit_scored,
            "impostor_rejected": impostor_rejected, "impostor_scored": impostor_scored,
            "far": far, "frr": frr, "acc": acc, "auc": auc, "eer": eer,
            "eer_thr": eer_thr,
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

    micro_frr = 1 - tla / tls       if tls > 0 else 0.0
    micro_far = (tis - tir) / tis   if tis > 0 else 0.0
    micro_acc = (tla + tir) / (tls + tis) if (tls + tis) > 0 else 0.0

    print(f"\n{'=' * 70}")
    print(f"  Aggregate Results ({len(all_results)} users)")
    print(f"{'=' * 70}")
    print(f"  {'User':<12} {'FAR':>8} {'FRR':>8} {'Accuracy':>10} {'AUC':>8} {'EER':>8}")
    print(f"  {'-' * 58}")
    for r in all_results:
        print(f"  {r['user']:<12} {r['far']*100:>7.1f}% "
              f"{r['frr']*100:>7.1f}% {r['acc']*100:>9.1f}% "
              f"{r['auc']:>8.4f} {r['eer']*100:>7.1f}%")
    print(f"  {'-' * 58}")
    print(f"  {'Mean':<12} {mean_far*100:>7.1f}% "
          f"{mean_frr*100:>7.1f}% {mean_acc*100:>9.1f}% {mean_auc:>8.4f} "
          f"{sum(r['eer'] for r in all_results)/len(all_results)*100:>7.1f}%")
    print(f"  {'Micro':<12} {micro_far*100:>7.1f}% "
          f"{micro_frr*100:>7.1f}% {micro_acc*100:>9.1f}% {micro_auc:>8.4f}")
    print(f"\n  Total legitimate: {tla}/{tls} accepted")
    print(f"  Total impostors:  {tir}/{tis} rejected")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()