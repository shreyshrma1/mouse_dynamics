"""
separability_bank.py

Same diagnostics as diagnose_separability.py but for a single specified
user from the banking dataset, using Balabit users as impostors.

For the target user, computes:
  1. Intra-user variance  — mean pairwise distance within training windows
  2. Inter-user distance  — mean distance from user centroid to each Balabit impostor centroid
  3. Separability ratio   — inter / intra (higher = more separable)
  4. PCA explained variance — how many components needed for 90% variance
  5. t-SNE plot           — 2D visualization vs Balabit impostors

Outputs:
  - Console metrics
  - t-SNE plot saved to diagnostics/<user_id>_bank_tsne.png
  - Separability bar chart (single bar) saved to diagnostics/<user_id>_bank_sep.png

Usage:
    python measurements/separability_bank.py
    python measurements/separability_bank.py --window_size 10 --n_impostors 5
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_scroll import extract_session_features

BANK_DIR     = "bank_collection/bank-data"
BALABIT_DIR  = "balabit_dataset/training_files"
BALABIT_USERS = [
    "user7", "user9", "user12", "user15", "user16",
    "user20", "user21", "user23", "user29", "user35",
]

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
    parser.add_argument("--window_size",  type=int, default=10)
    parser.add_argument("--n_impostors",  type=int, default=5,
                        help="Number of Balabit impostors to show in t-SNE (default: 5)")
    parser.add_argument("--out_dir",      default="diagnostics")
    return parser.parse_args()


def get_session_files(user_dir):
    if not os.path.isdir(user_dir):
        return []
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def extract_windows(session_files, user_id, window_size):
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
        except Exception:
            pass
    return np.array(all_vecs) if all_vecs else np.empty((0, len(FEATURE_COLS)))


# ── Metrics ────────────────────────────────────────────────────────────────────

def mean_pairwise_distance(X):
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
    pca = PCA()
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cumvar, threshold) + 1)
    return n, pca.explained_variance_ratio_[:5]


def separability_ratio(user_vecs, impostor_vecs_list):
    if len(user_vecs) < 2:
        return float('nan'), float('nan'), float('nan')
    intra = mean_pairwise_distance(user_vecs)
    user_centroid = user_vecs.mean(axis=0)
    inter_dists = []
    for imp_vecs in impostor_vecs_list:
        if len(imp_vecs) > 0:
            inter_dists.append(np.linalg.norm(user_centroid - imp_vecs.mean(axis=0)))
    inter = float(np.mean(inter_dists)) if inter_dists else float('nan')
    ratio = inter / intra if intra > 0 else float('nan')
    return intra, inter, ratio


# ── t-SNE plot ─────────────────────────────────────────────────────────────────

def tsne_plot(user_id, user_train, user_test, impostor_dict, out_path):
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
    perplexity = min(30, n - 1)
    if n < 4:
        print(f"  [!] Not enough windows for t-SNE ({n}), skipping")
        return

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=1000, verbose=0)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    color_map = {
        'user_train': ('#00d4ff', 'o', 100, f'{user_id} (train)'),
        'user_test':  ('#00ff88', 's', 100, f'{user_id} (test)'),
    }
    imp_colors = ['#ff6b6b', '#ffa726', '#ce93d8', '#80cbc4', '#fff176',
                  '#ef9a9a', '#a5d6a7', '#90caf9', '#ffcc02', '#f48fb1']
    for i, imp_user in enumerate(impostor_dict.keys()):
        color_map[f'imp_{imp_user}'] = (
            imp_colors[i % len(imp_colors)], '^', 60, f'Balabit {imp_user}'
        )

    unique_labels = list(dict.fromkeys(labels))
    for lbl in unique_labels:
        mask = [l == lbl for l in labels]
        pts  = X_2d[mask]
        color, marker, size, name = color_map.get(lbl, ('#888888', 'x', 40, lbl))
        alpha = 0.9 if lbl.startswith('user') else 0.5
        ax.scatter(pts[:, 0], pts[:, 1], c=color, marker=marker,
                   s=size, label=name, alpha=alpha, edgecolors='none')

    ax.set_title(f't-SNE: {user_id} (bank) vs Balabit impostors',
                 color='white', fontsize=13, pad=12)
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.legend(framealpha=0.3, facecolor='#111133', labelcolor='white', fontsize=8)
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

    user_id = input("Enter bank user ID: ").strip()

    user_dir = os.path.join(BANK_DIR, user_id)
    if not os.path.isdir(user_dir):
        print(f"No data directory found at {user_dir}")
        sys.exit(1)

    print(f"\nWindow size: {args.window_size}  |  Features: {len(FEATURE_COLS)}  |  "
          f"Impostors in t-SNE: {args.n_impostors}\n")

    # Load target user
    print(f"Loading windows for {user_id}...")
    user_files = get_session_files(user_dir)
    user_vecs  = extract_windows(user_files, user_id, args.window_size)
    print(f"  {user_id}: {len(user_vecs)} windows")

    if len(user_vecs) < 8:
        print(f"Not enough windows ({len(user_vecs)}) — collect more data")
        sys.exit(1)

    # Load Balabit impostors
    print("Loading Balabit impostor windows...")
    balabit_vecs = {}
    for b_user in BALABIT_USERS:
        files = get_session_files(os.path.join(BALABIT_DIR, b_user))
        vecs  = extract_windows(files, b_user, args.window_size)
        balabit_vecs[b_user] = vecs
        print(f"  {b_user}: {len(vecs)} windows")

    # Train/test split
    n_total = len(user_vecs)
    n_test  = max(1, int(n_total * 0.25))
    n_train = n_total - n_test
    train_vecs = user_vecs[:n_train]
    test_vecs  = user_vecs[n_train:]

    # Standardize: fit on target user training windows only
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vecs)
    test_scaled  = scaler.transform(test_vecs)
    impostor_scaled = {
        u: scaler.transform(v) if len(v) > 0 else v
        for u, v in balabit_vecs.items()
    }

    # Separability
    impostor_list = [impostor_scaled[u] for u in BALABIT_USERS
                     if u in impostor_scaled and len(impostor_scaled[u]) > 0]
    intra, inter, ratio = separability_ratio(train_scaled, impostor_list)

    # PCA
    n_components, top5_var = pca_components_for_variance(train_scaled, threshold=0.90)
    top5_str = ', '.join(f'{v*100:.1f}%' for v in top5_var)

    print(f"\n{'=' * 50}")
    print(f"User: {user_id} (banking data)")
    print(f"{'=' * 50}")
    print(f"  Windows: {n_total} total  |  Train: {n_train}  |  Test: {n_test}")
    print(f"  Intra-user mean pairwise dist: {intra:.4f}")
    print(f"  Inter-user centroid dist:      {inter:.4f}  (vs Balabit users)")
    print(f"  Separability ratio (inter/intra): {ratio:.4f}")
    print(f"  PCA components for 90% variance: {n_components}/{len(FEATURE_COLS)}")
    print(f"  Top-5 PC explained variance: [{top5_str}]")

    if ratio >= 2.0:
        verdict = "Good — user is well-separated from Balabit impostors"
    elif ratio >= 1.5:
        verdict = "Marginal — some overlap with Balabit impostors likely"
    else:
        verdict = "Poor — user overlaps heavily with Balabit impostors"
    print(f"  Verdict: {verdict}")

    # t-SNE
    imp_trimmed = {}
    for u, v in list(impostor_scaled.items())[:args.n_impostors]:
        imp_trimmed[u] = v[:40]

    tsne_out = os.path.join(args.out_dir, f"{user_id}_bank_tsne.png")
    tsne_plot(user_id, train_scaled[:60], test_scaled, imp_trimmed, tsne_out)

    # Bar chart
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    color = '#00d4ff' if ratio >= 2.0 else '#ffa726' if ratio >= 1.5 else '#ff6b6b'
    ax.bar([user_id], [ratio], color=color, edgecolor='none')
    ax.axhline(y=1.5, color='#ffffff44', linestyle='--', linewidth=1,
               label='Ratio = 1.5 (marginal)')
    ax.axhline(y=2.0, color='#ffffff88', linestyle='--', linewidth=1,
               label='Ratio = 2.0 (good)')
    ax.text(0, ratio + 0.05, f'{ratio:.2f}', ha='center', va='bottom',
            color='white', fontsize=11)
    ax.set_title(f'Separability: {user_id} vs Balabit', color='white',
                 fontsize=12, pad=10)
    ax.set_ylabel('Separability Ratio', color='#888888')
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.legend(framealpha=0.3, facecolor='#111133', labelcolor='white', fontsize=9)

    plt.tight_layout()
    sep_out = os.path.join(args.out_dir, f"{user_id}_bank_sep.png")
    plt.savefig(sep_out, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Separability chart saved → {sep_out}")


if __name__ == "__main__":
    main()