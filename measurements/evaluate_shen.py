"""
evaluate_shen.py

Evaluate a trained Shen pipeline model against:
  - Held-out legitimate windows (saved during training) -> should be ACCEPTED
  - Impostor sessions (balabit_dataset/)               -> should be REJECTED

Usage:
    python measurements/evaluate_shen.py
"""

import sys
import os
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_sess import extract_session_features

SAVE_DIR     = "checkpoints_shen"
IMPOSTOR_DIR = "balabit_dataset/training_files"

FEATURE_COLS = [
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


def get_session_files(directory):
    if not os.path.isdir(directory):
        return []
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ])


def extract_windows(session_files, user_id, window_size):
    all_vecs = []
    for path in session_files:
        try:
            df = extract_session_features(path, user_id, window_size=window_size)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
            if len(df) == 0:
                continue
            for _, grp in df.groupby("session"):
                rows = grp[FEATURE_COLS].values
                if len(rows) >= 1:
                    all_vecs.append(rows.mean(axis=0))
        except Exception as e:
            print(f"  [!] {os.path.basename(path)}: {e}")
    return all_vecs


def score_vecs(vecs, model, reference, dist_mean, dist_std):
    if len(vecs) == 0:
        return np.array([])
    x = np.abs(np.array(vecs) - reference)
    x = (x - dist_mean) / np.where(dist_std < 1e-9, 1.0, dist_std)
    return model.decision_function(x)


def main():
    user = input("Enter user ID: ").strip()

    save_path = os.path.join(SAVE_DIR, user)
    try:
        model = joblib.load(os.path.join(save_path, "model.pkl"))
        state = joblib.load(os.path.join(save_path, "state.pkl"))
    except FileNotFoundError:
        print(f"No trained model found at {save_path} — run train_from_existing_shen.py first")
        sys.exit(1)

    reference    = state["reference"]
    dist_mean    = state["dist_mean"]
    dist_std     = state["dist_std"]
    test_samples = state["test_samples"]
    window_size  = state["window_size"]

    print(f"\nLoaded model for '{user}' "
          f"(train={state['n_train']} windows, held-out={state['n_test']} windows, "
          f"nu={state['nu']}, gamma={state['gamma']})\n")

    # ── Legitimate: held-out windows ──────────────────────────────────────
    legit_scores   = score_vecs(test_samples, model, reference, dist_mean, dist_std)
    legit_accepted = int(np.sum(legit_scores >= 0))
    legit_scored   = len(legit_scores)

    print(f"Legitimate held-out ({legit_scored} windows):")
    if legit_scored > 0:
        print(f"  Scores: min={legit_scores.min():+.4f}, "
              f"mean={legit_scores.mean():+.4f}, max={legit_scores.max():+.4f}")
        print(f"  Accepted: {legit_accepted}/{legit_scored}")

    # ── Impostors ─────────────────────────────────────────────────────────
    impostor_accepted = 0
    impostor_scored   = 0
    all_impostor_scores = []

    impostor_users = sorted(os.listdir(IMPOSTOR_DIR))
    print(f"\nImpostors ({len(impostor_users)} users):")
    for imp_user in impostor_users:
        imp_files  = get_session_files(os.path.join(IMPOSTOR_DIR, imp_user))
        imp_vecs   = extract_windows(imp_files, imp_user, window_size)
        imp_scores = score_vecs(imp_vecs, model, reference, dist_mean, dist_std)
        if len(imp_scores) == 0:
            continue
        accepted = int(np.sum(imp_scores >= 0))
        impostor_accepted       += accepted
        impostor_scored         += len(imp_scores)
        all_impostor_scores.extend(imp_scores.tolist())
        print(f"  {imp_user:<12} {len(imp_scores) - accepted}/{len(imp_scores)} rejected")

    impostor_rejected = impostor_scored - impostor_accepted

    # ── AUC ───────────────────────────────────────────────────────────────
    all_scores = np.concatenate([legit_scores, np.array(all_impostor_scores)])
    all_labels = np.concatenate([
        np.ones(len(legit_scores)),   # 1 = legitimate
        np.zeros(len(all_impostor_scores))  # 0 = impostor
    ])
    auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) == 2 else float("nan")

    # ── Summary ───────────────────────────────────────────────────────────
    frr = 1 - legit_accepted    / legit_scored    if legit_scored    > 0 else 0.0
    far = impostor_accepted / impostor_scored      if impostor_scored > 0 else 0.0
    acc = (legit_accepted + impostor_rejected) / (legit_scored + impostor_scored) \
          if (legit_scored + impostor_scored) > 0 else 0.0

    print(f"""
        {'=' * 45}
        Results for '{user}'
        {'=' * 45}
        Legitimate:  {legit_accepted}/{legit_scored} accepted   FRR = {frr*100:.1f}%
        Impostors:   {impostor_rejected}/{impostor_scored} rejected  FAR = {far*100:.1f}%
        Accuracy:    {acc*100:.1f}%
        AUC:         {auc:.4f}
        {'=' * 45}
        """)


if __name__ == "__main__":
    main()