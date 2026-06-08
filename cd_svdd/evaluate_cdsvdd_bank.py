import sys
import os
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_scroll import extract_session_features

SAVE_DIR     = "checkpoints_cdsvdd_bank"
IMPOSTOR_DIR = "balabit_dataset/training_files"


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
    user = input("Enter user ID: ").strip()

    save_path = os.path.join(SAVE_DIR, user)
    try:
        model = joblib.load(os.path.join(save_path, "model.pkl"))
        state = joblib.load(os.path.join(save_path, "state.pkl"))
    except FileNotFoundError:
        print(f"No trained model found at {save_path} — run train_cdsvdd_bank.py first")
        sys.exit(1)

    scaler       = state["scaler"]
    test_samples = state["test_samples"]
    window_size  = state["window_size"]
    feature_cols = state["feature_cols"]
    more_scroll  = state.get("more_scroll", False)
    dir_scroll   = state.get("dir_scroll",  False)

    print(f"\nLoaded model for '{user}' "
          f"(train={state['n_train']} windows, held-out={state['n_test']} windows, "
          f"nu={state['nu']})\n")

    # ── Legitimate: held-out windows ──────────────────────────────────────
    test_norm      = scaler.transform(test_samples)
    legit_scores   = model.score(test_norm)
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
        imp_files  = get_session_files(os.path.join(IMPOSTOR_DIR, imp_user))
        imp_vecs   = extract_windows(imp_files, imp_user, window_size, feature_cols,
                                     more_scroll=more_scroll, dir_scroll=dir_scroll)
        if len(imp_vecs) == 0:
            continue
        imp_norm   = scaler.transform(np.array(imp_vecs, dtype=np.float32))
        imp_scores = model.score(imp_norm)
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