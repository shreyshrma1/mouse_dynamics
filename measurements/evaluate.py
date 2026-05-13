"""
evaluate.py

Evaluate the ContinualTrainer39 model against:
  - Legitimate sessions (collected_data/<user>/) → should be ACCEPTED
  - Impostor sessions (all Balabit users)        → should be REJECTED

Reports accuracy, FAR, and FRR.

Usage:
    python measurements/evaluate.py \
        --user user1 \
        --legit_dir collected_data \
        --impostor_dir balabit_dataset/training_files
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.continual_trainer import ContinualTrainer


def get_session_files(directory):
    """Return all files in a directory regardless of extension."""
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user",         required=True,
                        help="User ID whose checkpoint to load")
    parser.add_argument("--legit_dir",    default="collected_data",
                        help="Root directory containing collected_data/<user>/")
    parser.add_argument("--impostor_dir", required=True,
                        help="Root directory containing per-user Balabit subdirectories")
    parser.add_argument("--save_dir",     default="checkpoints_ocsvm")
    parser.add_argument("--verbose",      action="store_true",
                        help="Print individual session scores")
    args = parser.parse_args()

    # load model
    trainer = ContinualTrainer(user_id=args.user, save_dir=args.save_dir)
    trainer.load()

    if not trainer.is_ready:
        print("Model not yet trained — collect more sessions first.")
        return

    print(f"\n--- Enrollment Status for '{args.user}' ---")
    print(trainer.enrollment_status)
    print("-------------------------------------------\n")

    # ── Legitimate sessions ───────────────────────────────────────────────
    legit_dir = os.path.join(args.legit_dir, args.user)
    if not os.path.isdir(legit_dir):
        print(f"No legitimate data found at {legit_dir}")
        return

    legit_files = sorted([
        os.path.join(legit_dir, f)
        for f in os.listdir(legit_dir)
        if f.endswith(".csv")
    ])

    legit_accepted = 0
    legit_scored   = 0

    print(f"Scoring {len(legit_files)} legitimate sessions...")
    for path in legit_files:
        margin, accepted = trainer.score(path)
        if margin is None:
            continue
        legit_scored += 1
        if accepted:
            legit_accepted += 1
        if args.verbose:
            fname = os.path.basename(path)
            status = "✓ ACCEPTED" if accepted else "✗ REJECTED"
            print(f"  {fname:<45} {margin:>+8.4f}  {status}")

    # ── Impostor sessions ─────────────────────────────────────────────────
    impostor_dirs = sorted([
        os.path.join(args.impostor_dir, d)
        for d in os.listdir(args.impostor_dir)
        if os.path.isdir(os.path.join(args.impostor_dir, d))
    ])

    impostor_rejected = 0
    impostor_scored   = 0

    print(f"\nScoring impostor sessions from {len(impostor_dirs)} users...")
    for user_dir in impostor_dirs:
        user_name = os.path.basename(user_dir)
        files = get_session_files(user_dir)
        user_rejected = 0
        user_scored   = 0

        for path in files:
            margin, accepted = trainer.score(path)
            if margin is None:
                continue
            user_scored   += 1
            impostor_scored += 1
            if not accepted:
                user_rejected   += 1
                impostor_rejected += 1
            if args.verbose:
                fname = os.path.basename(path)
                status = "✓ REJECTED" if not accepted else "✗ ACCEPTED"
                print(f"  [{user_name}] {fname:<40} {margin:>+8.4f}  {status}")

        if user_scored > 0:
            pct = 100 * user_rejected / user_scored
            print(f"  {user_name:<10} {user_rejected}/{user_scored} rejected ({pct:.1f}%)")

    # ── Summary ───────────────────────────────────────────────────────────
    total_scored  = legit_scored + impostor_scored
    total_correct = legit_accepted + impostor_rejected

    frr = 1 - (legit_accepted / legit_scored)       if legit_scored     > 0 else 0.0
    far = 1 - (impostor_rejected / impostor_scored) if impostor_scored  > 0 else 0.0
    acc = total_correct / total_scored               if total_scored     > 0 else 0.0

    print(f"""
    {'='*55}
    Results
    {'='*55}
    Legitimate:  {legit_accepted}/{legit_scored} accepted
                FRR = {frr*100:.1f}% (legitimate sessions wrongly rejected)

    Impostors:   {impostor_rejected}/{impostor_scored} rejected
                FAR = {far*100:.1f}% (impostor sessions wrongly accepted)

    Overall:     {total_correct}/{total_scored} correct
                Accuracy = {acc*100:.1f}%
    {'='*55}
    """)


if __name__ == "__main__":
        main()