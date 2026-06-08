"""
shen_progression.py

Runs the OCSVM pipeline at four feature counts — 3, 8, 15, 48 — in a
single pass and prints a side-by-side comparison table.

Feature subsets are the top-N features from permutation importance
(feat_importance.py), ranked by mean AUC drop:

  1  num_critical_points
  2  num_points
  3  sum_of_angles        ← top-3 cutoff
  4  scroll_rate
  5  scroll_burst_len_mean
  6  sd_omega
  7  scroll_count
  8  straightness         ← top-8 cutoff
  9  max_vx
  10 sd_curv
  11 scroll_ratio
  12 max_a
  13 mean_omega
  14 min_vx
  15 min_vy              ← top-15 cutoff
  ... (all 48)           ← full feature set

Usage:
    python measurements/shen_progression.py
    python measurements/shen_progression.py --nu 0.06 --gamma scale
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
    "scroll_count", "scroll_rate", "scroll_ratio", "scroll_up_ratio",
    "scroll_dur_mean", "scroll_dur_std",
    "scroll_burst_count", "scroll_burst_dur_mean", "scroll_burst_len_mean",
]

ALL_FEATURE_COLS = HOLISTIC_COLS + SCROLL_COLS

# Full importance-ranked list (top 15 from feat_importance.py; rest appended)
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
    # remainder in original importance order (ranks 16-48)
    "scroll_dur_mean",
    "sd_jerk",
    "max_v",
    "mean_vx",
    "max_jerk",
    "mean_vy",
    "sd_vy",
    "mean_jerk",
    "sd_v",
    "scroll_burst_count",
    "largest_deviation",
    "max_vy",
    "a_beg_time",
    "sd_a",
    "elapsed_time",
    "min_jerk",
    "direction_of_movement",
    "scroll_up_ratio",
    "mean_v",
    "sd_vx",
    "min_a",
    "scroll_dur_std",
    "max_omega",
    "scroll_burst_dur_mean",
    "mean_a",
    "traveled_distance_pixel",
    "mean_curv",
    "type_of_action",
    "dist_end_to_end_line",
    "min_curv",
    "min_v",
    "min_omega",
    "max_curv",
]

PROGRESSIONS = [3, 8, 15, 48]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu",            type=float, default=0.06)
    parser.add_argument("--gamma",         default="scale")
    parser.add_argument("--window_size",   type=int,   default=50)
    parser.add_argument("--held_out_frac", type=float, default=0.25)
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_session_files(user_dir):
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def extract_all_windows(session_files, user_id, window_size):
    """Extract full 48-feature windows; subsetting happens later."""
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


def run_ocsvm(train_raw, X_test, y_test, nu, gamma):
    """Fit and evaluate OCSVM on already-subsetted arrays. Returns (auc, far, frr)."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)

    reference  = find_reference(train_scaled)
    train_dist = distance_vectors(train_scaled, reference)
    dist_mean  = train_dist.mean(axis=0)
    dist_std   = train_dist.std(axis=0)
    train_norm = normalize(train_dist, dist_mean, dist_std)

    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(train_norm)

    # Score test set
    x = scaler.transform(X_test)
    d = distance_vectors(x, reference)
    n = normalize(d, dist_mean, dist_std)
    scores = model.decision_function(n)

    auc = roc_auc_score(y_test, scores) if len(np.unique(y_test)) == 2 else float("nan")

    legit_mask = y_test == 1
    imp_mask   = y_test == 0
    far = (scores[imp_mask] >= 0).mean() if imp_mask.any() else 0.0
    frr = (scores[legit_mask] < 0).mean() if legit_mask.any() else 0.0

    return auc, far, frr, scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    gamma = args.gamma
    try:
        gamma = float(gamma)
    except ValueError:
        pass

    print(f"nu={args.nu}, gamma={gamma}, window_size={args.window_size}, "
          f"held_out_frac={args.held_out_frac}")
    print(f"Progressions: {PROGRESSIONS} features\n")

    # Pre-compute column indices for each progression
    indices = {
        k: [ALL_FEATURE_COLS.index(f) for f in RANKED_FEATURES[:k]]
        for k in PROGRESSIONS
    }

    # results[n_feat] = list of per-user dicts
    results = {k: [] for k in PROGRESSIONS}

    for user in BALABIT_USERS:
        print(f"── {user} ──")

        user_dir  = os.path.join(DATA_DIR, user)
        all_files = get_session_files(user_dir)
        all_vecs  = extract_all_windows(all_files, user, args.window_size)

        if len(all_vecs) < 8:
            print(f"  Not enough windows ({len(all_vecs)}), skipping\n")
            continue

        n_total = len(all_vecs)
        n_test  = max(1, int(n_total * args.held_out_frac))
        n_train = n_total - n_test

        full_train = np.array(all_vecs[:n_train])   # (n_train, 48)
        full_test_legit = np.array(all_vecs[n_train:])

        # Collect impostor windows (full 48 features)
        imp_vecs = []
        for other in BALABIT_USERS:
            if other == user:
                continue
            other_files = get_session_files(os.path.join(DATA_DIR, other))
            imp_vecs.extend(extract_all_windows(other_files, other, args.window_size))

        if not imp_vecs:
            print("  No impostor windows, skipping\n")
            continue

        full_test_imp = np.array(imp_vecs)

        # Build full test set once; subset per progression
        full_X_test = np.vstack([full_test_legit, full_test_imp])
        y_test = np.concatenate([
            np.ones(len(full_test_legit)),
            np.zeros(len(full_test_imp))
        ])

        for k in PROGRESSIONS:
            idx = indices[k]
            auc, far, frr, _ = run_ocsvm(
                full_train[:, idx],
                full_X_test[:, idx],
                y_test,
                args.nu, gamma
            )
            results[k].append({"user": user, "auc": auc, "far": far, "frr": frr})
            print(f"  top-{k:>2}:  AUC={auc:.4f}  FAR={far*100:.1f}%  FRR={frr*100:.1f}%")

        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    w = 12
    header = f"  {'User':<{w}}" + "".join(
        f"{'top-'+str(k)+' AUC':>{w}}{'FAR':>{w//2}}{'FRR':>{w//2}}"
        for k in PROGRESSIONS
    )
    sep = "=" * (w + len(PROGRESSIONS) * w * 2)

    print(f"\n{sep}")
    print(f"  Feature Progression Comparison")
    print(sep)
    print(header)
    print(f"  {'-' * (len(sep) - 4)}")

    # Collect all users that appear in all progressions
    all_users = [r["user"] for r in results[PROGRESSIONS[0]]]
    for user in all_users:
        row = f"  {user:<{w}}"
        for k in PROGRESSIONS:
            r = next((x for x in results[k] if x["user"] == user), None)
            if r:
                row += f"{r['auc']:>{w}.4f}{r['far']*100:>{w//2}.1f}%{r['frr']*100:>{w//2}.1f}%"
            else:
                row += f"{'—':>{w}}{'—':>{w//2}}{'—':>{w//2}}"
        print(row)

    print(f"  {'-' * (len(sep) - 4)}")

    # Mean row
    mean_row = f"  {'Mean':<{w}}"
    for k in PROGRESSIONS:
        rs = results[k]
        if rs:
            m_auc = np.mean([r["auc"] for r in rs if not np.isnan(r["auc"])])
            m_far = np.mean([r["far"] for r in rs])
            m_frr = np.mean([r["frr"] for r in rs])
            mean_row += f"{m_auc:>{w}.4f}{m_far*100:>{w//2}.1f}%{m_frr*100:>{w//2}.1f}%"
    print(mean_row)
    print(sep)

    # Column headers reminder
    print(f"\n  Columns per feature count: AUC | FAR | FRR")
    print(f"  Feature counts tested: {PROGRESSIONS}")


if __name__ == "__main__":
    main()