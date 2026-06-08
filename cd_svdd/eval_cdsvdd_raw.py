import sys
import os
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_raw_windows import extract_raw_windows

SAVE_DIR     = "checkpoints_cdsvdd_bank_raw"
IMPOSTOR_DIR = "balabit_dataset/training_files"


def get_session_files(directory):
    if not os.path.isdir(directory):
        return []
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ])


def main():
    user = input("Enter user ID: ").strip()

    save_path = os.path.join(SAVE_DIR, user)
    try:
        model = joblib.load(os.path.join(save_path, "model.pkl"))
        state = joblib.load(os.path.join(save_path, "state.pkl"))
    except FileNotFoundError:
        print(f"No trained model found at {save_path} — run train_cdsvdd_bank_raw.py first")
        sys.exit(1)

    test_windows = state["test_windows"]
    window_size  = state["window_size"]
    stride       = state["stride"]

    print(f"\nLoaded model for '{user}' "
          f"(train={state['n_train']} windows, held-out={state['n_test']} windows, "
          f"nu={state['nu']})\n")

    # ── Legitimate: held-out windows ──────────────────────────────────────
    legit_scores   = model.score(test_windows)
    legit_accepted = int(np.sum(legit_scores <= 0))
    legit_scored   = len(legit_scores)

    print(f"Legitimate held-out ({legit_scored} windows):")
    if legit_scored > 0:
        print(f"  Scores: min={legit_scores.min():+.4f}  "
              f"mean={legit_scores.mean():+.4f}  max={legit_scores.max():+.4f}")
        print(f"  Accepted: {legit_accepted}/{legit_scored}")

    # ── Impostors ─────────────────────────────────────────────────────────
    impostor_accepted   = 0
    impostor_scored     = 0
    all_impostor_scores = []

    impostor_users = sorted(os.listdir(IMPOSTOR_DIR))
    print(f"\nImpostors ({len(impostor_users)} users):")
    for imp_user in impostor_users:
        imp_files   = get_session_files(os.path.join(IMPOSTOR_DIR, imp_user))
        imp_windows = extract_raw_windows(imp_files, window_size, stride)
        if len(imp_windows) == 0:
            continue
        imp_arr    = np.array(imp_windows, dtype=np.float32)
        imp_scores = model.score(imp_arr)
        accepted   = int(np.sum(imp_scores <= 0))
        impostor_accepted       += accepted
        impostor_scored         += len(imp_scores)
        all_impostor_scores.extend(imp_scores.tolist())
        print(f"  {imp_user:<12} {len(imp_scores) - accepted}/{len(imp_scores)} rejected")

    impostor_rejected = impostor_scored - impostor_accepted

    # ── Metrics ───────────────────────────────────────────────────────────
    all_scores = np.concatenate([legit_scores, np.array(all_impostor_scores)])
    all_labels = np.concatenate([
        np.ones(len(legit_scores)),
        np.zeros(len(all_impostor_scores)),
    ])
    auc = roc_auc_score(all_labels, -all_scores) \
          if len(np.unique(all_labels)) == 2 else float("nan")

    frr = 1 - legit_accepted / legit_scored       if legit_scored    > 0 else 0.0
    far = impostor_accepted  / impostor_scored     if impostor_scored > 0 else 0.0
    acc = (legit_accepted + impostor_rejected) / (legit_scored + impostor_scored) \
          if (legit_scored + impostor_scored) > 0 else 0.0

    print(f"""
{'='*45}
Results for '{user}'
{'='*45}
Legitimate:  {legit_accepted}/{legit_scored} accepted   FRR = {frr*100:.1f}%
Impostors:   {impostor_rejected}/{impostor_scored} rejected  FAR = {far*100:.1f}%
Accuracy:    {acc*100:.1f}%
AUC:         {auc:.4f}
{'='*45}
""")


if __name__ == "__main__":
    main()