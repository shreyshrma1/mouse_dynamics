import sys
import os
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_raw_windows import extract_raw_windows, INPUT_DIM
from backbone_tcn import TCNNetwork
from cd_svdd_model import CDSVDD

DATA_DIR      = "bank_collection/bank-data"
SAVE_DIR      = "checkpoints_cdsvdd_bank_raw"
WINDOW_SIZE   = 100
STRIDE        = 100
NU            = 0.1
N_EPOCHS      = 50
LR            = 1e-3
CHANNELS      = 32
N_LAYERS      = 4
OUTPUT_DIM    = 8
QP_SUBSAMPLE  = 256
HELD_OUT_FRAC = 0.25


def get_session_files(user_dir):
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def main():
    user_id = input("Enter user ID: ").strip()

    user_dir = os.path.join(DATA_DIR, user_id)
    if not os.path.isdir(user_dir):
        print(f"No data directory found at {user_dir}")
        sys.exit(1)

    session_files = get_session_files(user_dir)
    print(f"Found {len(session_files)} session files for {user_id}")

    all_windows = extract_raw_windows(session_files, WINDOW_SIZE, STRIDE)
    if len(all_windows) < 8:
        print(f"Not enough windows ({len(all_windows)}) to train — collect more data")
        sys.exit(1)

    n_total = len(all_windows)
    n_test  = max(1, int(n_total * HELD_OUT_FRAC))
    n_train = n_total - n_test

    train_windows = np.array(all_windows[:n_train], dtype=np.float32)
    test_windows  = np.array(all_windows[n_train:],  dtype=np.float32)

    print(f"Total windows: {n_total}  |  Train: {n_train}  |  Held-out: {n_test}")

    effective_qp = min(QP_SUBSAMPLE, n_train)

    net = TCNNetwork(
        input_dim=INPUT_DIM,
        channels=CHANNELS,
        n_layers=N_LAYERS,
        output_dim=OUTPUT_DIM,
    )

    model = CDSVDD(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        nu=NU,
        n_epochs=N_EPOCHS,
        lr=LR,
        qp_subsample=effective_qp,
    )
    model.net = net
    model.fit(train_windows)

    save_path = os.path.join(SAVE_DIR, user_id)
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(model, os.path.join(save_path, "model.pkl"))
    joblib.dump({
        "test_windows": test_windows,
        "n_train":      n_train,
        "n_test":       n_test,
        "nu":           NU,
        "window_size":  WINDOW_SIZE,
        "stride":       STRIDE,
    }, os.path.join(save_path, "state.pkl"))

    print(f"Model saved to {save_path}/")


if __name__ == "__main__":
    main()