"""
evaluate_shen_scroll.py

Loads a trained OCSVM checkpoint and evaluates it against held-out
legitimate windows and impostor session files.

Defaults match the original bank collection behaviour:
    save_dir     = checkpoints_shen_scroll_bank
    impostor_dir = balabit_dataset/training_files

To evaluate a model trained on collected_data against Balabit impostors:
    python measurements/evaluate_shen_scroll.py
        --save_dir checkpoints_shen_scroll_collected

To evaluate against another bank user instead of Balabit:
    python measurements/evaluate_shen_scroll.py
        --impostor_dir bank_collection/bank-data

Usage:
    python measurements/evaluate_shen_scroll.py
    python measurements/evaluate_shen_scroll.py
        --save_dir checkpoints_shen_scroll_collected
    python measurements/evaluate_shen_scroll.py
        --save_dir checkpoints_shen_scroll_collected
        --impostor_dir bank_collection/bank-data
"""

import sys
import os
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_scroll import extract_session_features, MORE_SCROLL_COLS, DIR_SCROLL_COLS

DEFAULT_SAVE_DIR     = "checkpoints_shen_scroll_bank"
DEFAULT_IMPOSTOR_DIR = "balabit_dataset/training_files"


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR,
                        help=f"Directory containing model checkpoints "
                             f"(default: {DEFAULT_SAVE_DIR})")
    parser.add_argument("--impostor_dir", type=str, default=DEFAULT_IMPOSTOR_DIR,
                        help=f"Root directory containing per-user impostor session folders "
                             f"(default: {DEFAULT_IMPOSTOR_DIR})")
    return parser.parse_args()


def get_session_files(directory):
    if not os.path.isdir(directory):
        return []
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ])


def extract_windows(session_files, user_id, window_size, feature_cols,
                    more_scroll=False, dir_scroll=False):
    all_vecs = []
    for path in session_files:
        try:
            df = extract_session_features(path, user_id, window_size=window_size,
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


def score_vecs(vecs, model, scaler, reference, dist_mean, dist_std):
    if len(vecs) == 0:
        return np.array([])
    x = scaler.transform(np.array(vecs))
    x = np.abs(x - reference)
    x = (x - dist_mean) / np.where(dist_std < 1e-9, 1.0, dist_std)
    return model.decision_function(x)


def main():
    args = parse_args()
    save_dir     = args.save_dir
    impostor_dir = args.impostor_dir

    print(f"[Config] save_dir={save_dir}  impostor_dir={impostor_dir}")

    user = input("Enter user ID: ").strip()

    save_path = os.path.join(save_dir, user)
    try:
        model = joblib.load(os.path.join(save_path, "model.pkl"))
        state = joblib.load(os.path.join(save_path, "state.pkl"))
    except FileNotFoundError:
        print(f"No trained model found at {save_path} — run train_from_existing_shen_scroll.py first")
        sys.exit(1)

    reference    = state["reference"]
    dist_mean    = state["dist_mean"]
    dist_std     = state["dist_std"]
    scaler       = state["scaler"]
    test_samples = state["test_samples"]
    window_size  = state["window_size"]
    feature_cols = state["feature_cols"]
    more_scroll  = state.get("more_scroll", False)
    dir_scroll   = state.get("dir_scroll",  False)
    top_n        = state.get("top_n", None)

    feature_desc = f"top-{top_n}" if top_n else "all"
    print(f"\nLoaded model for '{user}' "
          f"(train={state['n_train']} windows, held-out={state['n_test']} windows, "
          f"nu={state['nu']}, gamma={state['gamma']}, "
          f"features={feature_desc} [{len(feature_cols)}])\n")

    # ── Legitimate: held-out windows ──────────────────────────────────────
    legit_scores   = score_vecs(test_samples, model, scaler, reference, dist_mean, dist_std)
    legit_accepted = int(np.sum(legit_scores >= 0))
    legit_scored   = len(legit_scores)

    print(f"Legitimate held-out ({legit_scored} windows):")
    if legit_scored > 0:
        print(f"  Scores: min={legit_scores.min():+.4f}, "
              f"mean={legit_scores.mean():+.4f}, max={legit_scores.max():+.4f}")
        print(f"  Accepted: {legit_accepted}/{legit_scored}")

    # ── Impostors ─────────────────────────────────────────────────────────
    if not os.path.isdir(impostor_dir):
        print(f"\n[Error] Impostor directory not found: {impostor_dir}")
        sys.exit(1)

    impostor_accepted   = 0
    impostor_scored     = 0
    all_impostor_scores = []

    impostor_users = sorted(os.listdir(impostor_dir))
    # Skip the target user if they happen to be in the impostor directory
    impostor_users = [u for u in impostor_users if u != user]

    print(f"\nImpostors ({len(impostor_users)} users from {impostor_dir}):")
    for imp_user in impostor_users:
        imp_dir    = os.path.join(impostor_dir, imp_user)
        if not os.path.isdir(imp_dir):
            continue
        imp_files  = get_session_files(imp_dir)
        imp_vecs   = extract_windows(imp_files, imp_user, window_size, feature_cols,
                                     more_scroll=more_scroll, dir_scroll=dir_scroll)
        imp_scores = score_vecs(imp_vecs, model, scaler, reference, dist_mean, dist_std)
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
        np.ones(len(legit_scores)),
        np.zeros(len(all_impostor_scores))
    ])
    auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) == 2 else float("nan")

    frr = 1 - legit_accepted / legit_scored       if legit_scored    > 0 else 0.0
    far = impostor_accepted  / impostor_scored     if impostor_scored > 0 else 0.0
    acc = (legit_accepted + impostor_rejected) / (legit_scored + impostor_scored) \
          if (legit_scored + impostor_scored) > 0 else 0.0

    print(f"""
{'=' * 45}
Results for '{user}'  (features: {feature_desc})
{'=' * 45}
Legitimate:  {legit_accepted}/{legit_scored} accepted   FRR = {frr*100:.1f}%
Impostors:   {impostor_rejected}/{impostor_scored} rejected  FAR = {far*100:.1f}%
Accuracy:    {acc*100:.1f}%
AUC:         {auc:.4f}
{'=' * 45}
""")


if __name__ == "__main__":
    main()