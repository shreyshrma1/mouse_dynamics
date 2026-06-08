"""
diagnose_separability.py

Diagnostic tool to understand why certain users are hard to authenticate.

For each Balabit user, computes:
  1. Intra-user variance  — mean pairwise distance within training windows
  2. Inter-user distance  — mean distance from user centroid to each impostor centroid
  3. Separability ratio   — inter / intra (higher = more separable)
  4. PCA explained variance — how many components needed for 90% variance
  5. t-SNE plot           — 2D visualization of user vs impostor windows in
                            neighborhood-preserving space

Outputs:
  - Console table of separability metrics per user
  - One t-SNE plot per user saved to diagnostics/<user>_tsne.png
  - A summary plot of separability ratios saved to diagnostics/separability.png

Usage:
    python measurements/diagnose_separability.py
    python measurements/diagnose_separability.py --window_size 50 --n_impostors 3
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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
    "scroll_count","scroll_rate","scroll_ratio","scroll_up_ratio",
    "scroll_dur_mean","scroll_dur_std",
    "scroll_burst_count","scroll_burst_dur_mean","scroll_burst_len_mean",
]

FEATURE_COLS = HOLISTIC_COLS + SCROLL_COLS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size",  type=int, default=50)
    parser.add_argument("--n_impostors",  type=int, default=3,
                        help="Number of impostor users to include in t-SNE (default: 3)")
    parser.add_argument("--out_dir",      default="diagnostics",
                        help="Output directory for plots (default: diagnostics)")
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
            df = df.replace([float('inf'), float('-inf')], float('nan'))
            df = df.dropna(subset=FEATURE_COLS)
            if len(df) == 0:
                continue
            for _, grp in df.groupby("session"):
                rows = grp[FEATURE_COLS].values
                if len(rows) >= 1:
                    all_vecs.append(rows.mean(axis=0))
        except Exception as e:
            pass
    return np.array(all_vecs) if all_vecs else np.empty((0, len(FEATURE_COLS)))


# ── Metrics ────────────────────────────────────────────────────────────────────

def mean_pairwise_distance(X):
    """Mean L2 distance between all pairs of rows in X."""
    if len(X) < 2:
        return 0.0
    n = len(X)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(X[i] - X[j])
            count += 1
    return total / count if count > 0 else 0.0


def pca_components_for_variance(X, threshold=0.90):
    """Number of PCA components needed to explain `threshold` of variance."""
    pca = PCA()
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cumvar, threshold) + 1)
    return n, pca.explained_variance_ratio_[:5]  # also return top-5 ratios


def separability_ratio(user_vecs, all_impostor_vecs_list):
    """
    inter / intra separability ratio.
    intra = mean pairwise distance within user's windows
    inter = mean distance from user centroid to each impostor centroid
    """
    if len(user_vecs) < 2:
        return float('nan'), float('nan'), float('nan')

    intra = mean_pairwise_distance(user_vecs)
    user_centroid = user_vecs.mean(axis=0)

    inter_dists = []
    for imp_vecs in all_impostor_vecs_list:
        if len(imp_vecs) > 0:
            imp_centroid = imp_vecs.mean(axis=0)
            inter_dists.append(np.linalg.norm(user_centroid - imp_centroid))

    inter = float(np.mean(inter_dists)) if inter_dists else float('nan')
    ratio = inter / intra if intra > 0 else float('nan')
    return intra, inter, ratio


# ── t-SNE plot ─────────────────────────────────────────────────────────────────

def tsne_plot(user, user_train, user_test, impostor_dict, out_path):
    """
    Plot t-SNE of user training windows, user test windows, and a sample
    of impostor windows. Saves to out_path.
    """
    labels = []
    all_vecs = []

    all_vecs.append(user_train)
    labels.extend(['user_train'] * len(user_train))

    if len(user_test) > 0:
        all_vecs.append(user_test)
        labels.extend(['user_test'] * len(user_test))

    for imp_user, imp_vecs in impostor_dict.items():
        if len(imp_vecs) > 0:
            all_vecs.append(imp_vecs)
            labels.extend([f'imp_{imp_user}'] * len(imp_vecs))

    X = np.vstack(all_vecs)
    n = len(X)

    # t-SNE needs at least 4 samples and perplexity < n
    perplexity = min(30, n - 1)
    if n < 4:
        print(f"  [!] Not enough windows for t-SNE ({n}), skipping plot")
        return

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=1000, verbose=0)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    color_map = {
        'user_train': ('#00d4ff', 'o', 100, 'User (train)'),
        'user_test':  ('#00ff88', 's', 100, 'User (test)'),
    }
    imp_colors = ['#ff6b6b', '#ffa726', '#ce93d8', '#80cbc4', '#fff176',
                  '#ef9a9a', '#a5d6a7', '#90caf9', '#ffcc02', '#f48fb1']

    imp_users = list(impostor_dict.keys())
    for i, imp_user in enumerate(imp_users):
        color_map[f'imp_{imp_user}'] = (
            imp_colors[i % len(imp_colors)], '^', 60, f'Impostor {imp_user}'
        )

    unique_labels = list(dict.fromkeys(labels))
    for lbl in unique_labels:
        mask = [l == lbl for l in labels]
        pts  = X_2d[mask]
        color, marker, size, name = color_map.get(lbl, ('#888888', 'x', 40, lbl))
        alpha = 0.9 if lbl.startswith('user') else 0.5
        ax.scatter(pts[:, 0], pts[:, 1], c=color, marker=marker,
                   s=size, label=name, alpha=alpha, edgecolors='none')

    ax.set_title(f't-SNE: {user} vs impostors', color='white', fontsize=13, pad=12)
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    legend = ax.legend(framealpha=0.3, facecolor='#111133',
                       labelcolor='white', fontsize=8)
    ax.set_xlabel('t-SNE 1', color='#888888', fontsize=9)
    ax.set_ylabel('t-SNE 2', color='#888888', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved t-SNE plot → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Window size: {args.window_size}  |  "
          f"Features: {len(FEATURE_COLS)}  |  "
          f"Impostors in t-SNE: {args.n_impostors}\n")

    # Pre-load all users
    print("Loading windows for all users...")
    all_user_vecs = {}
    for user in BALABIT_USERS:
        files = get_session_files(os.path.join(DATA_DIR, user))
        vecs  = extract_all_windows(files, user, args.window_size)
        all_user_vecs[user] = vecs
        print(f"  {user}: {len(vecs)} windows")

    print()

    results = []

    for user in BALABIT_USERS:
        print(f"{'=' * 50}")
        print(f"User: {user}")
        print(f"{'=' * 50}")

        vecs = all_user_vecs[user]
        if len(vecs) < 8:
            print(f"  Not enough windows ({len(vecs)}), skipping\n")
            continue

        n_total = len(vecs)
        n_test  = max(1, int(n_total * 0.25))
        n_train = n_total - n_test

        train_vecs = vecs[:n_train]
        test_vecs  = vecs[n_train:]

        # ── Standardize: fit on training windows only ──────────────────────
        imp_sample = {
            u: all_user_vecs[u]
            for u in BALABIT_USERS
            if u != user and len(all_user_vecs[u]) > 0
        }
        scaler = StandardScaler()
        train_vecs_scaled = scaler.fit_transform(train_vecs)
        test_vecs_scaled  = scaler.transform(test_vecs)
        impostor_vecs_scaled = {
            u: scaler.transform(v) if len(v) > 0 else v
            for u, v in imp_sample.items()
        }

        # ── Separability (on scaled vecs) ─────────────────────────────────
        impostor_vecs_list_scaled = [
            impostor_vecs_scaled[u] for u in BALABIT_USERS
            if u != user and u in impostor_vecs_scaled
        ]
        intra, inter, ratio = separability_ratio(train_vecs_scaled,
                                                  impostor_vecs_list_scaled)

        # ── PCA explained variance (on scaled vecs) ───────────────────────
        n_components, top5_var = pca_components_for_variance(train_vecs_scaled,
                                                              threshold=0.90)
        top5_str = ', '.join(f'{v*100:.1f}%' for v in top5_var)

        print(f"  Windows: {n_total} total  |  Train: {n_train}  |  Test: {n_test}")
        print(f"  Intra-user mean pairwise dist: {intra:.4f}")
        print(f"  Inter-user centroid dist:      {inter:.4f}")
        print(f"  Separability ratio (inter/intra): {ratio:.4f}")
        print(f"  PCA components for 90% variance: {n_components}/{len(FEATURE_COLS)}")
        print(f"  Top-5 PC explained variance: [{top5_str}]")

        results.append({
            'user': user,
            'n_train': n_train,
            'intra': intra,
            'inter': inter,
            'ratio': ratio,
            'pca_90': n_components,
        })

        # ── t-SNE (on scaled vecs) ────────────────────────────────────────
        imp_sample_trimmed = {}
        for u, v in list(impostor_vecs_scaled.items())[:args.n_impostors]:
            imp_sample_trimmed[u] = v[:40]

        tsne_out = os.path.join(args.out_dir, f"{user}_tsne.png")
        tsne_plot(user, train_vecs_scaled[:60], test_vecs_scaled,
                  imp_sample_trimmed, tsne_out)
        print()

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print(f"  Summary")
    print(f"{'=' * 68}")
    print(f"  {'User':<10} {'N_train':>8} {'Intra':>10} {'Inter':>10} "
          f"{'Ratio':>8} {'PCA90':>7}")
    print(f"  {'-' * 56}")
    for r in sorted(results, key=lambda x: x['ratio'], reverse=True):
        print(f"  {r['user']:<10} {r['n_train']:>8} {r['intra']:>10.4f} "
              f"{r['inter']:>10.4f} {r['ratio']:>8.4f} {r['pca_90']:>7}")
    print(f"{'=' * 68}")
    print(f"\n  Higher ratio = more separable = easier to authenticate")
    print(f"  Ratio < 1.5 suggests the user's data overlaps heavily with impostors")

    # ── Separability bar chart ─────────────────────────────────────────────
    users  = [r['user'] for r in results]
    ratios = [r['ratio'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    colors = ['#00d4ff' if r >= 2.0 else '#ffa726' if r >= 1.5 else '#ff6b6b'
              for r in ratios]
    bars = ax.bar(users, ratios, color=colors, edgecolor='none')
    ax.axhline(y=1.5, color='#ffffff44', linestyle='--', linewidth=1,
               label='Ratio = 1.5 (marginal)')
    ax.axhline(y=2.0, color='#ffffff88', linestyle='--', linewidth=1,
               label='Ratio = 2.0 (good)')

    ax.set_title('Separability Ratio by User (inter/intra distance)',
                 color='white', fontsize=12, pad=10)
    ax.set_ylabel('Separability Ratio', color='#888888')
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    legend = ax.legend(framealpha=0.3, facecolor='#111133',
                       labelcolor='white', fontsize=9)

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{ratio:.2f}', ha='center', va='bottom',
                color='white', fontsize=8)

    plt.tight_layout()
    sep_out = os.path.join(args.out_dir, 'separability.png')
    plt.savefig(sep_out, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Separability chart saved → {sep_out}")


if __name__ == "__main__":
    main()