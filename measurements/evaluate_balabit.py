"""
evaluate_balabit.py

Evaluate each Balabit user's model against:
  - Their held-out sessions, which should be ACCEPTED
  - All other Balabit users' sessions, which should be REJECTED

Usage:
    python measurements/evaluate_balabit.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.continual_trainer_39 import ContinualTrainer39

BALABIT_USERS = [
    "user7", "user9", "user12", "user15", "user16",
    "user20", "user21", "user23", "user29", "user35",
]

DATA_DIR = "balabit_dataset/training_files"
SAVE_DIR = "checkpoints_ocsvm_balabit"


def get_session_files(user_dir):
    if not os.path.isdir(user_dir):
        return []
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def main():
    all_results = []

    for user in BALABIT_USERS:
        print(f"\n{'=' * 50}")
        print(f"Evaluating {user}")
        print(f"{'=' * 50}")

        trainer = ContinualTrainer39(user_id=user, save_dir=SAVE_DIR)
        trainer.load()

        if not trainer.is_ready:
            print(f"  No trained model found for {user}, skipping")
            continue

        held_out_path = os.path.join(SAVE_DIR, f"{user}_held_out.txt")
        if not os.path.exists(held_out_path):
            print(f"  No held-out file found at {held_out_path}, skipping")
            continue

        with open(held_out_path) as f:
            legit_files = [line.strip() for line in f if line.strip()]

        legit_accepted = 0
        legit_scored   = 0

        print("\n  Legitimate held-out sessions:")
        for path in legit_files:
            margin, accepted = trainer.score(path)
            if margin is None:
                print(f"  [legit]  {os.path.basename(path):<30} not scored")
                continue
            legit_scored   += 1
            legit_accepted += int(accepted)
            status = "ACCEPT" if accepted else "REJECT"
            marker = "✓" if accepted else "✗"
            print(
                f"  [legit]  {os.path.basename(path):<30} "
                f"{margin:>+8.4f}  {status:<6} {marker}"
            )

        impostor_rejected = 0
        impostor_scored   = 0

        print("\n  Impostor sessions:")
        for other_user in BALABIT_USERS:
            if other_user == user:
                continue
            for path in get_session_files(os.path.join(DATA_DIR, other_user)):
                margin, accepted = trainer.score(path)
                if margin is None:
                    continue
                impostor_scored   += 1
                impostor_rejected += int(not accepted)

        frr = 1 - (legit_accepted / legit_scored)       if legit_scored   > 0 else 0.0
        far = 1 - (impostor_rejected / impostor_scored) if impostor_scored > 0 else 0.0
        acc = (legit_accepted + impostor_rejected) / (legit_scored + impostor_scored) \
              if (legit_scored + impostor_scored) > 0 else 0.0

        print(f"\n  Legitimate: {legit_accepted}/{legit_scored} accepted")
        print(f"  Impostors:  {impostor_rejected}/{impostor_scored} rejected")
        print(f"  FAR: {far * 100:.1f}%   FRR: {frr * 100:.1f}%   Accuracy: {acc * 100:.1f}%")

        all_results.append({
            "user":              user,
            "legit_accepted":    legit_accepted,
            "legit_scored":      legit_scored,
            "impostor_rejected": impostor_rejected,
            "impostor_scored":   impostor_scored,
            "far": far,
            "frr": frr,
            "acc": acc,
        })

    if not all_results:
        print("No results to summarise.")
        return

    mean_far = sum(r["far"] for r in all_results) / len(all_results)
    mean_frr = sum(r["frr"] for r in all_results) / len(all_results)
    mean_acc = sum(r["acc"] for r in all_results) / len(all_results)

    total_legit_accepted    = sum(r["legit_accepted"]    for r in all_results)
    total_legit_scored      = sum(r["legit_scored"]      for r in all_results)
    total_impostor_rejected = sum(r["impostor_rejected"] for r in all_results)
    total_impostor_scored   = sum(r["impostor_scored"]   for r in all_results)

    micro_frr = 1 - total_legit_accepted    / total_legit_scored    if total_legit_scored    > 0 else 0.0
    micro_far = 1 - total_impostor_rejected / total_impostor_scored if total_impostor_scored > 0 else 0.0
    micro_acc = (
        (total_legit_accepted + total_impostor_rejected)
        / (total_legit_scored + total_impostor_scored)
        if (total_legit_scored + total_impostor_scored) > 0 else 0.0
    )

    print(f"\n{'=' * 55}")
    print(f"  Aggregate Results ({len(all_results)} users)")
    print(f"{'=' * 55}")
    print(f"  {'User':<12} {'FAR':>8} {'FRR':>8} {'Accuracy':>10}")
    print(f"  {'-' * 42}")
    for r in all_results:
        print(
            f"  {r['user']:<12} "
            f"{r['far'] * 100:>7.1f}% "
            f"{r['frr'] * 100:>7.1f}% "
            f"{r['acc'] * 100:>9.1f}%"
        )
    print(f"  {'-' * 42}")
    print(
        f"  {'Mean':<12} "
        f"{mean_far * 100:>7.1f}% "
        f"{mean_frr * 100:>7.1f}% "
        f"{mean_acc * 100:>9.1f}%"
    )
    print(
        f"  {'Micro':<12} "
        f"{micro_far * 100:>7.1f}% "
        f"{micro_frr * 100:>7.1f}% "
        f"{micro_acc * 100:>9.1f}%"
    )
    print(f"\n  Total legitimate: {total_legit_accepted}/{total_legit_scored} accepted")
    print(f"  Total impostors:  {total_impostor_rejected}/{total_impostor_scored} rejected")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()