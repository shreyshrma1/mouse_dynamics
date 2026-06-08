"""
feat_importance.py

Permutation feature importance for the OCSVM mouse dynamics model.
For each feature, shuffles its values across test windows and measures
the mean AUC drop. Large drop = feature matters; near-zero = noise.

Usage:
    python measurements/feat_importance.py
    python measurements/feat_importance.py --nu 0.06 --n_repeats 10
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

FEATURE_COLS = HOLISTIC_COLS + SCROLL_COLS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu",          type=float, default=0.06)
    parser.add_argument("--gamma",       default="scale")
    parser.add_argument("--window_size", type=int,   default=50)
    parser.add_argument("--held_out_frac", type=float, default=0.25)
    parser.add_argument("--n_repeats",   type=int,   default=10)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--out_dir",     default="diagnostics")
    return parser.parse_args()


# ── Helpers (identical to shen_replication_scroll.py) ────────────────────────

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
            df = df.replace([np.inf, -np.inf], float("nan")).dropna(subset=FEATURE_COLS)
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


def get_scores(X_raw, scaler, reference, dist_mean, dist_std, model):
    """Scale → distance → normalize → decision score."""
    if len(X_raw) == 0:
        return np.array([])
    x = scaler.transform(np.array(X_raw))
    d = distance_vectors(x, reference)
    n = normalize(d, dist_mean, dist_std)
    return model.decision_function(n)


# ── Permutation importance ────────────────────────────────────────────────────

def permutation_importance(X_test, y_test, scaler, reference,
                           dist_mean, dist_std, model, n_repeats, rng):
    """
    Returns array of shape (n_features,) with mean AUC drop per feature.
    Permutation happens in original (pre-scale) feature space.
    """
    baseline_scores = get_scores(X_test, scaler, reference, dist_mean, dist_std, model)
    baseline_auc = roc_auc_score(y_test, baseline_scores)

    n_features = X_test.shape[1]
    importance = np.zeros(n_features)

    for i in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            scores = get_scores(X_perm, scaler, reference, dist_mean, dist_std, model)
            drops.append(baseline_auc - roc_auc_score(y_test, scores))
        importance[i] = np.mean(drops)

    return importance


# ── Cumulative AUC ───────────────────────────────────────────────────────────

def cumulative_auc_curve(train_raw, X_test, y_test, feature_order,
                         nu, gamma,
                         reference_fn, distance_vectors_fn, normalize_fn):
    """
    Refit OCSVM adding one feature at a time (in importance order).
    Returns array of AUC values, one per feature count 1..n_features.
    """
    aucs = []
    n_features = len(feature_order)

    for k in range(1, n_features + 1):
        cols = list(feature_order[:k])

        # Refit scaler on the subset
        sc = StandardScaler()
        tr = sc.fit_transform(train_raw[:, cols])

        ref   = reference_fn(tr)
        tdist = distance_vectors_fn(tr, ref)
        dmean = tdist.mean(axis=0)
        dstd  = tdist.std(axis=0)
        tnorm = normalize_fn(tdist, dmean, dstd)

        mdl = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        mdl.fit(tnorm)

        X_sub = X_test[:, cols]
        x     = sc.transform(X_sub)
        d     = distance_vectors_fn(x, ref)
        n     = normalize_fn(d, dmean, dstd)
        scores = mdl.decision_function(n)
        aucs.append(roc_auc_score(y_test, scores))

    return np.array(aucs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    gamma = args.gamma
    try:
        gamma = float(gamma)
    except ValueError:
        pass

    print(f"nu={args.nu}, gamma={gamma}, window_size={args.window_size}, "
          f"n_repeats={args.n_repeats}, seed={args.seed}\n")

    # Collect per-user importance vectors and data for cumulative curve
    user_importances = []   # shape: (n_users, n_features)
    user_curve_data  = []   # list of (user_name, train_raw, X_test, y_test)

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

        train_raw = np.array(all_vecs[:n_train])
        test_legit = np.array(all_vecs[n_train:])

        # Collect impostor windows
        imp_vecs = []
        for other in BALABIT_USERS:
            if other == user:
                continue
            other_files = get_session_files(os.path.join(DATA_DIR, other))
            imp_vecs.extend(extract_all_windows(other_files, other, args.window_size))

        if len(imp_vecs) == 0:
            print("  No impostor windows, skipping\n")
            continue

        test_imp = np.array(imp_vecs)

        # Combine test set
        X_test = np.vstack([test_legit, test_imp])
        y_test = np.concatenate([
            np.ones(len(test_legit)),
            np.zeros(len(test_imp))
        ])

        # Fit pipeline (identical to shen_replication_scroll.py)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_raw)

        reference  = find_reference(train_scaled)
        train_dist = distance_vectors(train_scaled, reference)
        dist_mean  = train_dist.mean(axis=0)
        dist_std   = train_dist.std(axis=0)
        train_norm = normalize(train_dist, dist_mean, dist_std)

        model = OneClassSVM(kernel="rbf", nu=args.nu, gamma=gamma)
        model.fit(train_norm)

        # Baseline AUC
        baseline_scores = get_scores(X_test, scaler, reference, dist_mean, dist_std, model)
        baseline_auc = roc_auc_score(y_test, baseline_scores)
        print(f"  Baseline AUC: {baseline_auc:.4f}  |  "
              f"train={n_train}  legit_test={len(test_legit)}  imp_test={len(test_imp)}")

        # Permutation importance
        imp = permutation_importance(
            X_test, y_test, scaler, reference,
            dist_mean, dist_std, model, args.n_repeats, rng
        )
        user_importances.append(imp)
        user_curve_data.append((user, train_raw, X_test, y_test))
        print(f"  Done.\n")

    if not user_importances:
        print("No results.")
        return

    importance_matrix = np.array(user_importances)   # (n_users, n_features)
    mean_imp = importance_matrix.mean(axis=0)
    std_imp  = importance_matrix.std(axis=0)

    # ── Console table ─────────────────────────────────────────────────────────
    order = np.argsort(mean_imp)[::-1]

    print(f"\n{'=' * 62}")
    print(f"  Feature Importance  ({len(user_importances)} users, {args.n_repeats} repeats)")
    print(f"  Metric: mean AUC drop when feature is permuted")
    print(f"{'=' * 62}")
    print(f"  {'Rank':<6} {'Feature':<30} {'Mean Drop':>10} {'Std':>8}  Group")
    print(f"  {'-' * 58}")
    for rank, idx in enumerate(order, 1):
        fname = FEATURE_COLS[idx]
        group = "scroll" if fname in SCROLL_COLS else "holistic"
        print(f"  {rank:<6} {fname:<30} {mean_imp[idx]:>+10.4f} {std_imp[idx]:>8.4f}  {group}")

    # Group-level summary
    hol_idx = [i for i, f in enumerate(FEATURE_COLS) if f in HOLISTIC_COLS]
    scr_idx = [i for i, f in enumerate(FEATURE_COLS) if f in SCROLL_COLS]
    print(f"\n  Group summary:")
    print(f"    Holistic ({len(hol_idx)} features): mean drop = {mean_imp[hol_idx].mean():+.4f}")
    print(f"    Scroll   ({len(scr_idx)} features): mean drop = {mean_imp[scr_idx].mean():+.4f}")
    print(f"{'=' * 62}\n")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(6, len(FEATURE_COLS) * 0.28)))

    colors = ["steelblue" if FEATURE_COLS[i] in HOLISTIC_COLS else "darkorange"
              for i in order]

    y_pos = np.arange(len(order))
    ax.barh(y_pos, mean_imp[order], xerr=std_imp[order],
            color=colors, align="center", height=0.7,
            error_kw=dict(elinewidth=0.8, capsize=2))

    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEATURE_COLS[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean AUC drop (higher = more important)")
    ax.set_title(f"OCSVM Permutation Feature Importance\n"
                 f"({len(user_importances)} Balabit users, {args.n_repeats} repeats per feature)")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="steelblue",  label="Holistic"),
        Patch(facecolor="darkorange", label="Scroll"),
    ], loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "feature_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved to {out_path}")

    # ── Cumulative AUC curve ───────────────────────────────────────────────────
    print(f"\nComputing cumulative AUC curve ({len(user_curve_data)} users)...")

    all_curves = []
    for user, tr_raw, X_te, y_te in user_curve_data:
        print(f"  {user}...")
        curve = cumulative_auc_curve(
            tr_raw, X_te, y_te, order,
            args.nu, gamma,
            reference_fn=find_reference,
            distance_vectors_fn=distance_vectors,
            normalize_fn=normalize,
        )
        all_curves.append(curve)

    curves = np.array(all_curves)          # (n_users, n_features)
    mean_curve = curves.mean(axis=0)
    std_curve  = curves.std(axis=0)
    x_axis     = np.arange(1, len(order) + 1)

    # Find elbow: first point where AUC reaches 95% and 99% of its max
    max_auc = mean_curve.max()
    thresh_95 = next((k for k, v in enumerate(mean_curve) if v >= 0.95 * max_auc), len(order) - 1)
    thresh_99 = next((k for k, v in enumerate(mean_curve) if v >= 0.99 * max_auc), len(order) - 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_axis, mean_curve, color="steelblue", linewidth=2, label="Mean AUC")
    ax.fill_between(x_axis,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    alpha=0.2, color="steelblue", label="±1 std")

    # Mark thresholds
    for thresh, pct, color in [(thresh_95, "95%", "darkorange"),
                                (thresh_99, "99%", "crimson")]:
        ax.axvline(thresh + 1, color=color, linestyle="--", linewidth=1.2,
                   label=f"{pct} of max AUC @ {thresh + 1} features")
        ax.axhline(mean_curve[thresh], color=color, linestyle=":", linewidth=0.8)

    ax.set_xlabel("Number of features added (importance order)")
    ax.set_ylabel("Mean AUC across users")
    ax.set_title("Cumulative AUC vs. Feature Count\n(features added in permutation-importance order)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(order))

    plt.tight_layout()
    curve_path = os.path.join(args.out_dir, "cumulative_auc.png")
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Cumulative AUC chart saved to {curve_path}")

    # Print elbow summary
    print(f"\n  95% of max AUC ({max_auc:.4f}) reached at {thresh_95 + 1} features")
    print(f"  99% of max AUC reached at {thresh_99 + 1} features")
    print(f"  Top features at 95% threshold:")
    for idx in order[:thresh_95 + 1]:
        fname = FEATURE_COLS[idx]
        group = "scroll" if fname in SCROLL_COLS else "holistic"
        print(f"    {fname:<30} ({group})")


if __name__ == "__main__":
    main()