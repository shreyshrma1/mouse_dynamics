"""
train_nn_manhat.py

Identical to train_from_existing_shen_scroll.py but uses the Nearest
Neighbor (Manhattan) one-class detector instead of OCSVM.

The detector stores preprocessed training vectors directly — there is no
model fitting step. At scoring time, the average Manhattan distance to the
k nearest training vectors is computed and negated (higher = more legitimate).

The EER threshold is computed from the held-out test windows and saved to
state.pkl so evaluate_nn_manhat.py can apply it without needing impostor
data at evaluation time.

Saves to checkpoints_nn_manhat_bank/<user_id>/:
    state.pkl  - training vectors, preprocessing stats, threshold, held-out windows

Usage:
    python data_collection/train_nn_manhat.py
    python data_collection/train_nn_manhat.py --top_n 15
    python data_collection/train_nn_manhat.py
        --data_dir collected_data --save_dir checkpoints_nn_manhat_collected
"""

print("importing sys and os")
import sys
import os
print("importing joblib")
import joblib
print("importing numpy")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("importing extract_features_scroll")
from measurements.extract_features_scroll import extract_session_features, MORE_SCROLL_COLS, DIR_SCROLL_COLS
print("done importing extract_features_scroll")
print("importing sklearn")
from sklearn.preprocessing import StandardScaler
print("done importing sklearn")

DEFAULT_DATA_DIR = "bank_collection/bank-data"
DEFAULT_SAVE_DIR = "checkpoints_nn_manhat_bank"
WINDOW_SIZE      = 5
HELD_OUT_FRAC    = 0.25
K                = 3      # k nearest neighbours, as per Shen et al. (2014)

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

ALL_FEATURE_COLS = HOLISTIC_COLS + SCROLL_COLS

RANKED_FEATURES = [
    "num_critical_points","num_points","sum_of_angles","scroll_rate",
    "scroll_burst_len_mean","sd_omega","scroll_count","straightness",
    "max_vx","sd_curv","scroll_ratio","max_a","mean_omega","min_vx","min_vy",
]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--top_n",    type=int, default=None,
                        help="Use only the top-N ranked features (1-15). "
                             "Omit to use all 48 features.")
    parser.add_argument("--k",        type=int, default=K,
                        help=f"Number of nearest neighbours (default: {K})")
    parser.add_argument("--more_scroll", action="store_true")
    parser.add_argument("--dir_scroll",  action="store_true")
    return parser.parse_args()


def get_session_files(user_dir):
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def extract_all_windows(session_files, user_id, feature_cols,
                        more_scroll=False, dir_scroll=False):
    all_vecs = []
    for path in session_files:
        print(f"  Processing {os.path.basename(path)}...")
        try:
            df = extract_session_features(path, user_id, window_size=WINDOW_SIZE,
                                          more_scroll=more_scroll, dir_scroll=dir_scroll)
            if df.empty or not all(c in df.columns for c in feature_cols):
                continue
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
            if len(df) == 0:
                continue
            for _, grp in df.groupby("session"):
                rows = grp[feature_cols].values
                if len(rows) >= 1:
                    all_vecs.append(rows.mean(axis=0))
        except Exception as e:
            print(f"  [!] {os.path.basename(path)}: {e}")
    return all_vecs


def find_reference(train_samples):
    n = len(train_samples)
    print(f"Running find_reference on {n} samples...")
    mean_dists = np.zeros(n)
    for i in range(n):
        dists = np.sum(np.abs(train_samples - train_samples[i]), axis=1)
        mean_dists[i] = dists.sum() / max(n - 1, 1)
    print("Done.")
    return train_samples[np.argmin(mean_dists)]


def nn_manhattan_score(test_vecs, train_vecs, k):
    test_vecs = np.atleast_2d(test_vecs)
    scores    = np.zeros(len(test_vecs))
    for i, test_vec in enumerate(test_vecs):
        dists     = np.sum(np.abs(train_vecs - test_vec), axis=1)
        scores[i] = np.sort(dists)[:k].mean()
    return -scores


def main():
    args = parse_args()

    print(f"[Config] data_dir={args.data_dir}  save_dir={args.save_dir}  k={args.k}")

    if args.top_n is not None:
        top_n = max(1, min(args.top_n, len(RANKED_FEATURES)))
        feature_cols = RANKED_FEATURES[:top_n]
        print(f"Using top-{top_n} features: {feature_cols}")
    else:
        top_n = None
        feature_cols = (ALL_FEATURE_COLS
                        + (MORE_SCROLL_COLS if args.more_scroll else [])
                        + (DIR_SCROLL_COLS  if args.dir_scroll  else []))
        print(f"Using all {len(feature_cols)} features")

    user_id = input("Enter user ID: ").strip()

    user_dir = os.path.join(args.data_dir, user_id)
    if not os.path.isdir(user_dir):
        print(f"No data directory found at {user_dir}")
        sys.exit(1)

    session_files = get_session_files(user_dir)
    print(f"Found {len(session_files)} session files for {user_id}")

    all_vecs = extract_all_windows(session_files, user_id, feature_cols,
                                   args.more_scroll, args.dir_scroll)
    if len(all_vecs) < 8:
        print(f"Not enough windows ({len(all_vecs)}) to train — collect more data")
        sys.exit(1)

    n_total = len(all_vecs)
    n_test  = max(1, int(n_total * HELD_OUT_FRAC))
    n_train = n_total - n_test

    train_samples = np.array(all_vecs[:n_train])
    test_samples  = np.array(all_vecs[n_train:])

    print(f"Total windows: {n_total}  |  Train: {n_train}  |  Held-out: {n_test}")
    print(f"Feature vector: {len(feature_cols)} features")

    # Shen preprocessing — fit on training data only
    scaler       = StandardScaler()
    train_scaled = scaler.fit_transform(train_samples)
    reference    = find_reference(train_scaled)
    train_dists  = np.abs(train_scaled - reference)
    dist_mean    = train_dists.mean(axis=0)
    dist_std     = train_dists.std(axis=0)
    std_safe     = np.where(dist_std < 1e-9, 1.0, dist_std)
    train_norm   = (train_dists - dist_mean) / std_safe

    # Score training data to report distribution
    train_scores = nn_manhattan_score(train_norm, train_norm, k=args.k)
    print(f"Train scores: min={train_scores.min():.4f}, "
          f"mean={train_scores.mean():.4f}, max={train_scores.max():.4f}")

    # Compute a default threshold from held-out windows alone (self-threshold).
    # This is the mean training score minus one std — a conservative accept boundary.
    # The full EER threshold requires impostor data and is computed in eval_nn_manhat.py.
    test_scaled = scaler.transform(test_samples)
    test_dists  = np.abs(test_scaled - reference)
    test_norm   = (test_dists - dist_mean) / std_safe
    test_scores = nn_manhattan_score(test_norm, train_norm, k=args.k)
    default_threshold = float(test_scores.mean() - test_scores.std())
    print(f"Default threshold (mean-std of held-out): {default_threshold:.4f}")

    save_path = os.path.join(args.save_dir, user_id)
    os.makedirs(save_path, exist_ok=True)
    joblib.dump({
        "train_norm":          train_norm,
        "scaler":              scaler,
        "reference":           reference,
        "dist_mean":           dist_mean,
        "dist_std":            dist_std,
        "test_samples":        test_samples,
        "feature_cols":        feature_cols,
        "top_n":               top_n,
        "n_train":             n_train,
        "n_test":              n_test,
        "k":                   args.k,
        "window_size":         WINDOW_SIZE,
        "more_scroll":         args.more_scroll,
        "dir_scroll":          args.dir_scroll,
        "default_threshold":   default_threshold,
    }, os.path.join(save_path, "state.pkl"))

    print(f"Model saved to {save_path}/")


if __name__ == "__main__":
    main()