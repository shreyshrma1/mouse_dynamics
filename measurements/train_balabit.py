"""
train_balabit.py

Train a ContinualTrainer39 model for each Balabit user, holding out
the last 2 sessions per user for testing.

Usage:
    python measurements/train_balabit.py
    python measurements/train_balabit.py --nu 0.05 --gamma scale
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.continual_trainer_39 import ContinualTrainer39, MIN_WINDOWS

BALABIT_USERS = [
    "user7", "user9", "user12", "user15", "user16",
    "user20", "user21", "user23", "user29", "user35",
]

DATA_DIR = "balabit_dataset/training_files"
SAVE_DIR = "checkpoints_ocsvm_balabit"
HELD_OUT = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nu",
        type=float,
        default=0.05,
        help="OCSVM nu parameter (default: 0.05).",
    )
    parser.add_argument(
        "--gamma",
        default="scale",
        help="OCSVM gamma parameter (default: scale).",
    )
    return parser.parse_args()


def get_session_files(user_dir):
    """Return all session files in a user directory sorted by name."""
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def main():
    args = parse_args()

    gamma = args.gamma
    try:
        gamma = float(gamma)
    except ValueError:
        pass  # keep as string e.g. "scale"

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"nu={args.nu}, gamma={gamma}")

    for user in BALABIT_USERS:
        print(f"\n{'=' * 50}")
        print(f"Training model for {user}")
        print(f"{'=' * 50}")

        user_dir = os.path.join(DATA_DIR, user)
        if not os.path.isdir(user_dir):
            print(f"  No directory found at {user_dir}, skipping")
            continue

        all_files = get_session_files(user_dir)

        if len(all_files) <= HELD_OUT:
            print(f"  Not enough sessions for {user}; found {len(all_files)}")
            continue

        train_files = all_files[:-HELD_OUT]
        test_files  = all_files[-HELD_OUT:]

        print(f"  Total sessions: {len(all_files)}")
        print(f"  Train: {len(train_files)} sessions")
        print(f"  Test:  {len(test_files)} sessions held out")

        held_out_path = os.path.join(SAVE_DIR, f"{user}_held_out.txt")
        with open(held_out_path, "w") as f:
            for path in test_files:
                f.write(path + "\n")
        print(f"  Held-out sessions saved to {held_out_path}")

        trainer = ContinualTrainer39(
            user_id=user,
            save_dir=SAVE_DIR,
            nu=args.nu,
            gamma=gamma,
        )

        # start fresh
        trainer.buffer.clear()
        trainer.model      = None
        trainer.scaler     = None
        trainer.n_sessions = 0

        for path in train_files:
            print(f"  Extracting {os.path.basename(path)}")
            for w in trainer._extract(path):
                trainer.buffer.append(w)
            trainer.n_sessions += 1

        print(f"  Buffer: {len(trainer.buffer)} windows")

        if len(trainer.buffer) >= MIN_WINDOWS:
            trainer._retrain()
        else:
            print(
                f"  Not enough windows to train: "
                f"{len(trainer.buffer)} / {MIN_WINDOWS} — skipping {user}"
            )
            continue

        trainer._save()
        print("\n" + trainer.enrollment_status)


if __name__ == "__main__":
    main()