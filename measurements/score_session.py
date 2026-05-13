"""
score_session.py

Load a saved ContinualTrainer checkpoint and score session files.

Usage:
    # score a specific session file
    python score_session.py --user user1 --session collected_data/user1/session_123.csv

    # score all legitimate sessions for a user
    python score_session.py --user user1 --all

    # score impostor sessions from a balabit user directory
    python score_session.py --user user1 --all --impostor_dir balabit_dataset/training_files/user7
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.continual_trainer import ContinualTrainer


def score_sessions(trainer, sessions, label="legitimate", expected_accept=True):
    print(f"Scoring {len(sessions)} {label} session(s)...\n")
    print(f"{'Session':<45} {'Score':>8} {'Decision':>12} {'Correct':>8}")
    print("-" * 75)

    accepted_count = 0
    correct_count  = 0
    scored_count   = 0

    for session_path in sessions:
        fname = os.path.basename(session_path)
        score, accepted = trainer.score(session_path)
        if score is None:
            print(f"{fname:<45} {'N/A':>8} {'no features':>12} {'':>8}")
            continue
        decision = "ACCEPTED" if accepted else "REJECTED"
        correct  = "✓" if accepted == expected_accept else "✗"
        print(f"{fname:<45} {score:>+8.4f} {decision:>12} {correct:>8}")
        scored_count   += 1
        accepted_count += int(accepted)
        correct_count  += int(accepted == expected_accept)

    if scored_count > 0:
        print("-" * 75)
        if expected_accept:
            print(f"Legitimate: {accepted_count}/{scored_count} accepted "
                  f"({100*accepted_count/scored_count:.1f}%) — "
                  f"{correct_count}/{scored_count} correct decisions\n")
        else:
            rejected_count = scored_count - accepted_count
            print(f"Impostor:   {rejected_count}/{scored_count} rejected "
                  f"({100*rejected_count/scored_count:.1f}%) — "
                  f"{correct_count}/{scored_count} correct decisions\n")

    return accepted_count, scored_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user",         required=True,
                        help="User ID whose checkpoint to load")
    parser.add_argument("--session",      default=None,
                        help="Path to a single session CSV to score")
    parser.add_argument("--all",          action="store_true",
                        help="Score all sessions in collected_data/<user>/")
    parser.add_argument("--impostor_dir", default=None,
                        help="Directory of impostor session files (should be REJECTED)")
    parser.add_argument("--save_dir",     default="checkpoints_ocsvm",
                        help="Directory where checkpoints are saved")
    parser.add_argument("--data_dir",     default="collected_data",
                        help="Directory where session files are saved")
    args = parser.parse_args()

    # load trainer from checkpoint
    trainer = ContinualTrainer(user_id=args.user, save_dir=args.save_dir)
    trainer.load()

    print(f"\n--- Enrollment Status for '{args.user}' ---")
    print(trainer.enrollment_status)
    print("----------------------------------------\n")

    if not trainer.is_ready:
        print("Model is not yet trained — collect more sessions first.")
        return

    # ── Legitimate sessions ───────────────────────────────────────────────
    if args.all:
        user_dir = os.path.join(args.data_dir, args.user)
        if not os.path.isdir(user_dir):
            print(f"No data directory found at {user_dir}")
            return
        sessions = sorted([
            os.path.join(user_dir, f)
            for f in os.listdir(user_dir)
            if f.endswith(".csv")
        ])
        if not sessions:
            print(f"No session files found in {user_dir}")
        else:
            score_sessions(trainer, sessions,
                           label="legitimate", expected_accept=True)

    elif args.session:
        score_sessions(trainer, [args.session],
                       label="legitimate", expected_accept=True)

    # ── Impostor sessions ─────────────────────────────────────────────────
    if args.impostor_dir:
        if not os.path.isdir(args.impostor_dir):
            print(f"Impostor directory not found: {args.impostor_dir}")
        else:
            impostor_sessions = sorted([
                os.path.join(args.impostor_dir, f)
                for f in os.listdir(args.impostor_dir)
                if os.path.isfile(os.path.join(args.impostor_dir, f))
            ])
            if not impostor_sessions:
                print(f"No session files found in {args.impostor_dir}")
            else:
                score_sessions(trainer, impostor_sessions,
                               label="impostor", expected_accept=False)

    if not args.all and not args.session and not args.impostor_dir:
        print("Error: provide --session <path>, --all, or --impostor_dir <path>")


if __name__ == "__main__":
    main()