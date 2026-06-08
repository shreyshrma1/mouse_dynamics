"""
eval_nn_manhat_user.py

Identical to eval_nn_manhat.py but evaluates against a single specified
impostor user from the banking data directory instead of all Balabit users.

Usage:
    python measurements/eval_nn_manhat_user.py
    python measurements/eval_nn_manhat_user.py
        --save_dir checkpoints_nn_manhat_collected
        --impostor_dir collected_data
"""

import sys
import os
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_scroll import extract_session_features, MORE_SCROLL_COLS, DIR_SCROLL_COLS

DEFAULT_SAVE_DIR     = "checkpoints_nn_manhat_bank"
DEFAULT_IMPOSTOR_DIR = "bank_collection/bank-data"


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",     type=str,   default=DEFAULT_SAVE_DIR)
    parser.add_argument("--impostor_dir", type=str,   default=DEFAULT_IMPOSTOR_DIR)
    parser.add_argument("--threshold", action="store_true",
                        help="Use the EER threshold saved in state.pkl (computed by "
                             "eval_nn_manhat.py against Balabit). If omitted, the "
                             "optimal threshold is computed fresh from this impostor user.")
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


def preprocess(vecs, scaler, reference, dist_mean, dist_std):
    x = scaler.transform(np.array(vecs))
    x = np.abs(x - reference)
    x = (x - dist_mean) / np.where(dist_std < 1e-9, 1.0, dist_std)
    return x


def nn_manhattan_score(test_vecs, train_vecs, k):
    test_vecs = np.atleast_2d(test_vecs)
    scores    = np.zeros(len(test_vecs))
    for i, test_vec in enumerate(test_vecs):
        dists     = np.sum(np.abs(train_vecs - test_vec), axis=1)
        scores[i] = np.sort(dists)[:k].mean()
    return -scores


def find_eer_threshold(legit_scores, impostor_scores):
    all_scores  = np.concatenate([legit_scores, impostor_scores])
    thresholds  = np.unique(all_scores)
    best_thresh = thresholds[0]
    best_diff   = float("inf")
    for t in thresholds:
        frr  = np.mean(legit_scores    < t)
        far  = np.mean(impostor_scores >= t)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff   = diff
            best_thresh = t
    eer = (np.mean(legit_scores < best_thresh) +
           np.mean(impostor_scores >= best_thresh)) / 2
    return best_thresh, eer


def score_vecs(vecs, train_norm, scaler, reference, dist_mean, dist_std, k):
    if len(vecs) == 0:
        return np.array([])
    x = preprocess(vecs, scaler, reference, dist_mean, dist_std)
    return nn_manhattan_score(x, train_norm, k)


def main():
    args = parse_args()
    print(f"[Config] save_dir={args.save_dir}  impostor_dir={args.impostor_dir}")

    user = input("Enter user ID: ").strip()

    save_path = os.path.join(args.save_dir, user)
    try:
        state = joblib.load(os.path.join(save_path, "state.pkl"))
    except FileNotFoundError:
        print(f"No trained model found at {save_path} — run train_nn_manhat.py first")
        sys.exit(1)

    train_norm   = state["train_norm"]
    scaler       = state["scaler"]
    reference    = state["reference"]
    dist_mean    = state["dist_mean"]
    dist_std     = state["dist_std"]
    test_samples = state["test_samples"]
    window_size  = state["window_size"]
    feature_cols = state["feature_cols"]
    more_scroll  = state.get("more_scroll", False)
    dir_scroll   = state.get("dir_scroll",  False)
    top_n            = state.get("top_n", None)
    k                = state.get("k", 3)
    k                = state.get("k", 3)
    saved_threshold  = state.get("eer_threshold", state.get("default_threshold", None))

    feature_desc = f"top-{top_n}" if top_n else "all"
    print(f"\nLoaded model for '{user}' "
          f"(train={state['n_train']} windows, held-out={state['n_test']} windows, "
          f"k={k}, features={feature_desc} [{len(feature_cols)}])")

    # ── Impostor user selection ───────────────────────────────────────────
    if not os.path.isdir(args.impostor_dir):
        print(f"[Error] Impostor directory not found: {args.impostor_dir}")
        sys.exit(1)

    available = [u for u in sorted(os.listdir(args.impostor_dir))
                 if os.path.isdir(os.path.join(args.impostor_dir, u)) and u != user]
    if not available:
        print(f"[Error] No other users found in {args.impostor_dir}")
        sys.exit(1)

    print(f"Available impostor users: {', '.join(available)}")
    imp_user = input("Enter impostor user ID: ").strip()

    if imp_user not in available:
        print(f"[Error] '{imp_user}' not found in {args.impostor_dir}")
        sys.exit(1)

    # ── Legitimate: held-out windows ──────────────────────────────────────
    legit_scores = score_vecs(test_samples, train_norm, scaler,
                              reference, dist_mean, dist_std, k)
    legit_scored = len(legit_scores)

    # ── Impostor ──────────────────────────────────────────────────────────
    imp_dir    = os.path.join(args.impostor_dir, imp_user)
    imp_vecs   = extract_windows(get_session_files(imp_dir), imp_user,
                                 window_size, feature_cols, more_scroll, dir_scroll)
    imp_scores = score_vecs(imp_vecs, train_norm, scaler,
                            reference, dist_mean, dist_std, k)

    if len(imp_scores) == 0:
        print(f"[Error] No windows extracted for impostor '{imp_user}'")
        sys.exit(1)

    impostor_scored = len(imp_scores)

    # Threshold: use saved EER threshold or compute optimal from this impostor
    if args.threshold:
        if saved_threshold is None:
            print("[Warning] No saved threshold found in state.pkl — computing from impostor data.")
            threshold, eer = find_eer_threshold(legit_scores, imp_scores)
            threshold_source = "computed (no saved threshold found)"
        else:
            threshold = saved_threshold
            threshold_source = "saved (from eval_nn_manhat.py vs Balabit)"
            eer = (np.mean(legit_scores < threshold) +
                   np.mean(imp_scores >= threshold)) / 2
    else:
        threshold, eer = find_eer_threshold(legit_scores, imp_scores)
        threshold_source = "computed (optimal for this impostor)"

    print(f"Threshold: {threshold:.4f} ({threshold_source})")

    # Save per-user-pair threshold so bank_to_bank_1d.py can load it
    state[f"eer_threshold_{imp_user}"] = float(threshold)
    joblib.dump(state, os.path.join(save_path, "state.pkl"))

    legit_accepted    = int(np.sum(legit_scores >= threshold))
    impostor_accepted = int(np.sum(imp_scores   >= threshold))
    impostor_rejected = impostor_scored - impostor_accepted

    print(f"\nLegitimate held-out ({legit_scored} windows):")
    print(f"  Scores: min={legit_scores.min():+.4f}, "
          f"mean={legit_scores.mean():+.4f}, max={legit_scores.max():+.4f}")
    print(f"  Accepted: {legit_accepted}/{legit_scored}")

    print(f"\nImpostor '{imp_user}' ({impostor_scored} windows):")
    print(f"  Scores: min={imp_scores.min():+.4f}, "
          f"mean={imp_scores.mean():+.4f}, max={imp_scores.max():+.4f}")
    print(f"  Rejected: {impostor_rejected}/{impostor_scored}")

    # ── AUC ───────────────────────────────────────────────────────────────
    all_scores = np.concatenate([legit_scores, imp_scores])
    all_labels = np.concatenate([np.ones(legit_scored), np.zeros(impostor_scored)])
    auc = roc_auc_score(all_labels, all_scores) \
          if len(np.unique(all_labels)) == 2 else float("nan")

    frr = 1 - legit_accepted    / legit_scored    if legit_scored    > 0 else 0.0
    far = impostor_accepted      / impostor_scored if impostor_scored > 0 else 0.0
    acc = (legit_accepted + impostor_rejected) / (legit_scored + impostor_scored) \
          if (legit_scored + impostor_scored) > 0 else 0.0

    print(f"""
{'=' * 45}
Results for '{user}' vs '{imp_user}'  (features: {feature_desc}, k={k})
{'=' * 45}
EER threshold: {threshold:+.4f}   EER: {eer*100:.2f}%
Legitimate:  {legit_accepted}/{legit_scored} accepted   FRR = {frr*100:.1f}%
Impostor:    {impostor_rejected}/{impostor_scored} rejected  FAR = {far*100:.1f}%
Accuracy:    {acc*100:.1f}%
AUC:         {auc:.4f}
{'=' * 45}
""")


if __name__ == "__main__":
    main()