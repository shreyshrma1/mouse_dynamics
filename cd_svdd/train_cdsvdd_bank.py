import sys
import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_scroll import extract_session_features, MORE_SCROLL_COLS, DIR_SCROLL_COLS
from cd_svdd_model import CDSVDD

DATA_DIR      = "bank_collection/bank-data"
SAVE_DIR      = "checkpoints_cdsvdd_bank"
WINDOW_SIZE   = 10
NU            = 0.1
HELD_OUT_FRAC = 0.25
N_EPOCHS      = 50
LR            = 1e-3
HIDDEN_DIM    = 32
OUTPUT_DIM    = 16

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
    "mean_jerk","sd_jerk","max_jerk","min_jerk","a_beg_time",
]

SCROLL_COLS = [
    "scroll_count","scroll_rate","scroll_ratio","scroll_up_ratio",
    "scroll_dur_mean","scroll_dur_std",
    "scroll_burst_count","scroll_burst_dur_mean","scroll_burst_len_mean",
]


def get_session_files(user_dir):
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def extract_all_windows(session_files, user_id, feature_cols, more_scroll=False, dir_scroll=False):
    all_vecs = []
    for path in session_files:
        try:
            df = extract_session_features(path, user_id, window_size=WINDOW_SIZE,
                                          more_scroll=more_scroll, dir_scroll=dir_scroll)
            df = df.replace([float("inf"), float("-inf")], float("nan")).dropna(subset=feature_cols)
            if len(df) == 0:
                continue
            for _, grp in df.groupby("session"):
                rows = grp[feature_cols].values
                if len(rows) >= 1:
                    all_vecs.append(rows.mean(axis=0))
        except Exception as e:
            print(f"  [!] {os.path.basename(path)}: {e}")
    return all_vecs


def main():
    user_id     = input("Enter user ID: ").strip()
    more_scroll = "--more_scroll" in sys.argv
    dir_scroll  = "--dir_scroll"  in sys.argv

    feature_cols = (HOLISTIC_COLS + SCROLL_COLS
                    + (MORE_SCROLL_COLS if more_scroll else [])
                    + (DIR_SCROLL_COLS  if dir_scroll  else []))

    user_dir = os.path.join(DATA_DIR, user_id)
    if not os.path.isdir(user_dir):
        print(f"No data directory found at {user_dir}")
        sys.exit(1)

    session_files = get_session_files(user_dir)
    print(f"Found {len(session_files)} session files for {user_id}")

    all_vecs = extract_all_windows(session_files, user_id, feature_cols,
                                   more_scroll=more_scroll, dir_scroll=dir_scroll)
    if len(all_vecs) < 8:
        print(f"Not enough windows ({len(all_vecs)}) to train — collect more data")
        sys.exit(1)

    n_total = len(all_vecs)
    n_test  = max(1, int(n_total * HELD_OUT_FRAC))
    n_train = n_total - n_test

    train_samples = np.array(all_vecs[:n_train], dtype=np.float32)
    test_samples  = np.array(all_vecs[n_train:],  dtype=np.float32)

    print(f"Total windows: {n_total}  |  Train: {n_train}  |  Held-out: {n_test}")
    print(f"Feature vector: {len(feature_cols)} total")

    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_samples)

    model = CDSVDD(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        nu=NU,
        n_epochs=N_EPOCHS,
        lr=LR,
    )
    model.fit(train_norm)

    save_path = os.path.join(SAVE_DIR, user_id)
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(model,  os.path.join(save_path, "model.pkl"))
    joblib.dump({
        "scaler":       scaler,
        "test_samples": test_samples,
        "n_train":      n_train,
        "n_test":       n_test,
        "nu":           NU,
        "window_size":  WINDOW_SIZE,
        "feature_cols": feature_cols,
        "more_scroll":  more_scroll,
        "dir_scroll":   dir_scroll,
    }, os.path.join(save_path, "state.pkl"))

    print(f"Model saved to {save_path}/")


if __name__ == "__main__":
    main()