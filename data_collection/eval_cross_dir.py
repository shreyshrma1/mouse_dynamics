"""
eval_cross_dir.py

Evaluates user3's bank model against a user from a different data directory
whose raw session CSVs are extracted on the fly.

Since the external user has no trained model, only one direction is run:
  - Legitimate: user3's held-out test_samples (from checkpoint)
  - Impostor:   external user's windows extracted from raw session CSVs

This is useful for testing cross-collection-system generalization — e.g.
the same physical person recorded on two different machines/setups.

Usage:
    python eval_cross_dir.py --model_user user3 --ext_user user1
        --ext_dir collected_data
    python eval_cross_dir.py --model_user user3 --ext_user user1
        --ext_dir collected_data --runs 50
    python eval_cross_dir.py --model_user user3 --ext_user user1
        --ext_dir collected_data --runs 50 --fixed
"""

import os
import sys
import argparse
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants ─────────────────────────────────────────────────────────────────

N_SETS       = 20
SET_SIZE     = 10
N_LEGIT      = N_SETS // 2
N_IMPOSTOR   = N_SETS // 2
THRESHOLDS   = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_SEED = 42

CHECKPOINT_DIR = "checkpoints_shen_scroll_bank"
WINDOW_SIZE    = 5


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(user_id):
    path = os.path.join(CHECKPOINT_DIR, user_id)
    try:
        model = joblib.load(os.path.join(path, "model.pkl"))
        state = joblib.load(os.path.join(path, "state.pkl"))
    except FileNotFoundError:
        print(f"[Error] No model found at {path} — run train_from_existing_shen_scroll.py first.")
        sys.exit(1)

    print(f"[Model] Loaded for '{user_id}' — "
          f"train={state['n_train']} windows, held-out={state['n_test']} windows, "
          f"features={len(state['feature_cols'])}, nu={state['nu']}, gamma={state['gamma']}")

    return {
        "model":        model,
        "scaler":       state["scaler"],
        "reference":    state["reference"],
        "dist_mean":    state["dist_mean"],
        "dist_std":     state["dist_std"],
        "feature_cols": state["feature_cols"],
        "more_scroll":  state.get("more_scroll", False),
        "dir_scroll":   state.get("dir_scroll",  False),
        "test_samples": state["test_samples"],
    }


# ── External user window extraction ──────────────────────────────────────────

def get_session_files(user_dir):
    if not os.path.isdir(user_dir):
        return []
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def extract_windows_for_user(user_dir, user_id, feature_cols,
                              more_scroll=False, dir_scroll=False):
    """
    Extract all windows from raw session CSVs, preserving row order
    so contiguous slices are temporally meaningful.
    Returns np.ndarray of shape (n_windows, n_features).
    """
    from measurements.extract_features_scroll import extract_session_features

    session_files = get_session_files(user_dir)
    if not session_files:
        print(f"[Error] No session files found in {user_dir}")
        sys.exit(1)

    all_vecs = []
    for path in session_files:
        try:
            df = extract_session_features(
                path, user_id,
                window_size=WINDOW_SIZE,
                more_scroll=more_scroll,
                dir_scroll=dir_scroll,
            )
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

    return np.array(all_vecs) if all_vecs else np.empty((0, len(feature_cols)))


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(windows, m):
    """StandardScaler → Shen distance → normalize using model m's stats."""
    x = m["scaler"].transform(np.array(windows))
    x = np.abs(x - m["reference"])
    x = (x - m["dist_mean"]) / np.where(m["dist_std"] < 1e-9, 1.0, m["dist_std"])
    return x


# ── Set generation ────────────────────────────────────────────────────────────

def sample_contiguous_block(windows, rng):
    max_start = len(windows) - SET_SIZE
    start = int(rng.integers(0, max_start + 1))
    return windows[start:start + SET_SIZE], start


def generate_sets(target_user, legit_windows, impostor_user, impostor_windows, rng):
    sets = []
    for i in range(N_LEGIT):
        block, start = sample_contiguous_block(legit_windows, rng)
        sets.append({
            "set_id":  i + 1,
            "label":   "legitimate",
            "user":    target_user,
            "start":   start,
            "windows": block,
        })
    for i in range(N_IMPOSTOR):
        block, start = sample_contiguous_block(impostor_windows, rng)
        sets.append({
            "set_id":  N_LEGIT + i + 1,
            "label":   "impostor",
            "user":    impostor_user,
            "start":   start,
            "windows": block,
        })
    return sets


# ── Scoring & consensus ───────────────────────────────────────────────────────

def score_set(set_dict, m):
    x_norm = preprocess(set_dict["windows"], m)
    raw_scores      = m["model"].decision_function(x_norm)
    sample_accepted = [float(s) >= 0 for s in raw_scores]
    return raw_scores, sample_accepted


def consensus_verdict(sample_accepted, threshold):
    return (sum(sample_accepted) / len(sample_accepted)) >= threshold


def per_sample_accuracy(sample_accepted, label):
    correct = sum(sample_accepted) if label == "legitimate" else sum(not a for a in sample_accepted)
    return correct / len(sample_accepted)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results, threshold):
    legit_results    = [r for r in results if r["label"] == "legitimate"]
    impostor_results = [r for r in results if r["label"] == "impostor"]
    frr = sum(not r["consensus"][threshold] for r in legit_results) / len(legit_results) \
          if legit_results else 0.0
    far = sum(r["consensus"][threshold] for r in impostor_results) / len(impostor_results) \
          if impostor_results else 0.0
    return far, frr


def compute_sample_far_frr(results):
    legit_results    = [r for r in results if r["label"] == "legitimate"]
    impostor_results = [r for r in results if r["label"] == "impostor"]
    sample_frr = np.mean([
        sum(not a for a in r["sample_accepted"]) / len(r["sample_accepted"])
        for r in legit_results
    ]) if legit_results else 0.0
    sample_far = np.mean([
        sum(r["sample_accepted"]) / len(r["sample_accepted"])
        for r in impostor_results
    ]) if impostor_results else 0.0
    return sample_far, sample_frr


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model_user, model, ext_user, ext_windows, runs, base_seed, fixed):
    all_run_results = []
    legit_windows = model["test_samples"]

    for run_idx in range(runs):
        seed = (base_seed + run_idx) if fixed else None
        rng  = np.random.default_rng(seed)

        sets = generate_sets(model_user, legit_windows, ext_user, ext_windows, rng)

        results = []
        for s in sets:
            raw_scores, sample_accepted = score_set(s, model)
            sample_acc       = per_sample_accuracy(sample_accepted, s["label"])
            majority_verdict = consensus_verdict(sample_accepted, threshold=0.5)
            majority_correct = (majority_verdict == (s["label"] == "legitimate"))
            consensus         = {t: consensus_verdict(sample_accepted, t) for t in THRESHOLDS}
            consensus_correct = {
                t: (consensus[t] == (s["label"] == "legitimate")) for t in THRESHOLDS
            }
            results.append({
                **s,
                "raw_scores":        raw_scores,
                "sample_accepted":   sample_accepted,
                "sample_acc":        sample_acc,
                "majority_verdict":  majority_verdict,
                "majority_correct":  majority_correct,
                "consensus":         consensus,
                "consensus_correct": consensus_correct,
            })
        all_run_results.append(results)

    return all_run_results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary(all_run_results, runs, model_user, ext_user):
    sample_fars, sample_frrs = [], []
    thresh_fars = {t: [] for t in THRESHOLDS}
    thresh_frrs = {t: [] for t in THRESHOLDS}

    for results in all_run_results:
        sf, sr = compute_sample_far_frr(results)
        sample_fars.append(sf)
        sample_frrs.append(sr)
        for t in THRESHOLDS:
            far, frr = compute_metrics(results, t)
            thresh_fars[t].append(far)
            thresh_frrs[t].append(frr)

    print(f"\n{'═' * 60}")
    print(f"  Model: {model_user}  |  Legitimate: {model_user}  |  Impostor: {ext_user}")
    print(f"  {runs} run(s), {N_SETS} sets per run, {SET_SIZE} windows per set")
    print(f"{'═' * 60}")

    print(f"\nPer-window (before consensus):")
    print(f"  FAR : {np.mean(sample_fars):.1%}  (std {np.std(sample_fars):.1%})  "
          f"— ext user windows accepted by {model_user}'s model")
    print(f"  FRR : {np.mean(sample_frrs):.1%}  (std {np.std(sample_frrs):.1%})  "
          f"— {model_user}'s own windows rejected by their model")

    if runs > 1:
        print(f"\n  {'Threshold':<12} {'FAR mean':>10} {'FAR std':>9} {'FRR mean':>10} {'FRR std':>9}")
    else:
        print(f"\n  {'Threshold':<12} {'FAR':>8} {'FRR':>8}")

    for t in THRESHOLDS:
        marker = " ◄" if t == 0.5 else ""
        if runs > 1:
            print(f"  {t:<12.1f} "
                  f"{np.mean(thresh_fars[t]):>10.1%} {np.std(thresh_fars[t]):>9.1%} "
                  f"{np.mean(thresh_frrs[t]):>10.1%} {np.std(thresh_frrs[t]):>9.1%}{marker}")
        else:
            print(f"  {t:<12.1f} {thresh_fars[t][0]:>8.1%} {thresh_frrs[t][0]:>8.1%}{marker}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-directory continuous auth evaluation")
    parser.add_argument("--model_user", required=True,
                        help="User ID with a trained checkpoint in checkpoints_shen_scroll_bank/")
    parser.add_argument("--ext_user",   required=True,
                        help="External user ID whose raw CSVs will be extracted")
    parser.add_argument("--ext_dir",    required=True,
                        help="Root data directory containing ext_user's session folder "
                             "(e.g. collected_data). Script looks for <ext_dir>/<ext_user>/)")
    parser.add_argument("--runs",  type=int, default=1,
                        help="Number of evaluation runs to average (default: 1)")
    parser.add_argument("--fixed", action="store_true",
                        help="Use fixed RNG seeds for reproducibility")
    parser.add_argument("--seed",  type=int, default=DEFAULT_SEED,
                        help=f"Base RNG seed (only used with --fixed, default={DEFAULT_SEED})")
    args = parser.parse_args()

    print(f"[Config] model_user={args.model_user}  ext_user={args.ext_user}  "
          f"ext_dir={args.ext_dir}  Runs={args.runs}  "
          f"Fixed={'yes (seed=' + str(args.seed) + ')' if args.fixed else 'no'}")

    # Load bank model
    print()
    model = load_model(args.model_user)

    # Extract external user's windows using the model's feature cols
    ext_user_dir = os.path.join(args.ext_dir, args.ext_user)
    print(f"[Data] Extracting windows for '{args.ext_user}' from {ext_user_dir} ...")
    ext_windows = extract_windows_for_user(
        ext_user_dir, args.ext_user,
        model["feature_cols"],
        model["more_scroll"],
        model["dir_scroll"],
    )
    if len(ext_windows) < SET_SIZE:
        print(f"[Error] Only {len(ext_windows)} windows extracted for '{args.ext_user}' "
              f"— need at least {SET_SIZE}.")
        sys.exit(1)
    print(f"[Data] External user '{args.ext_user}': {len(ext_windows)} windows extracted")

    # Check legitimate side too
    legit_windows = model["test_samples"]
    if len(legit_windows) < SET_SIZE:
        print(f"[Error] '{args.model_user}' has only {len(legit_windows)} held-out windows "
              f"— need at least {SET_SIZE}.")
        sys.exit(1)
    print(f"[Data] Legitimate '{args.model_user}': {len(legit_windows)} held-out windows")

    # Run evaluation
    all_run_results = evaluate(
        args.model_user, model,
        args.ext_user, ext_windows,
        args.runs, args.seed, args.fixed,
    )

    print_summary(all_run_results, args.runs, args.model_user, args.ext_user)


if __name__ == "__main__":
    main()