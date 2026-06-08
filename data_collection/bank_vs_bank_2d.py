"""
bank_vs_bank_2d.py

Evaluates how well a user's OCSVM model distinguishes between themselves
and another user from the same banking application — a harder and more
realistic test than using Balabit impostors.

Both directions are run and aggregated:
  - User A's model scores blocks from user A (legitimate) and user B (impostor)
  - User B's model scores blocks from user B (legitimate) and user A (impostor)

For both the legitimate and impostor role, only held-out test windows
(saved in state.pkl) are used — no training data is ever evaluated.

Usage:
    python bank_vs_bank_2d.py --user_a user3 --user_b user5
    python bank_vs_bank_2d.py --user_a user3 --user_b user5 --runs 50
    python bank_vs_bank_2d.py --user_a user3 --user_b user5 --runs 50 --fixed
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
        "test_samples": state["test_samples"],   # raw feature space, pre-scaling
    }


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


# ── Single-direction evaluation ───────────────────────────────────────────────

def evaluate_direction(target_user, target_model, impostor_user, impostor_windows,
                       runs, base_seed, fixed):
    """
    Score target_user's held-out windows (legitimate) and impostor_user's
    held-out windows (impostor) against target_user's model.
    Returns list of run-result lists.
    """
    legit_windows = target_model["test_samples"]

    if len(legit_windows) < SET_SIZE:
        print(f"  [Skip] {target_user} has only {len(legit_windows)} held-out windows.")
        return []
    if len(impostor_windows) < SET_SIZE:
        print(f"  [Skip] {impostor_user} has only {len(impostor_windows)} held-out windows.")
        return []

    all_run_results = []
    for run_idx in range(runs):
        seed = (base_seed + run_idx) if fixed else None
        rng  = np.random.default_rng(seed)

        sets = generate_sets(target_user, legit_windows, impostor_user, impostor_windows, rng)

        results = []
        for s in sets:
            raw_scores, sample_accepted = score_set(s, target_model)
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

def print_direction_summary(label, all_run_results, runs):
    print(f"\n  {label}")
    print(f"  {'─' * 56}")

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

    print(f"  Per-window:  FAR={np.mean(sample_fars):.1%} (std {np.std(sample_fars):.1%})  "
          f"FRR={np.mean(sample_frrs):.1%} (std {np.std(sample_frrs):.1%})")

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

    return thresh_fars, thresh_frrs


def print_aggregate(thresh_fars_a, thresh_frrs_a, thresh_fars_b, thresh_frrs_b):
    """Average FAR/FRR across both directions."""
    print(f"\n{'═' * 60}")
    print(f"  Aggregate (both directions combined)")
    print(f"{'═' * 60}")
    print(f"\n  {'Threshold':<12} {'FAR mean':>10} {'FRR mean':>10}")
    for t in THRESHOLDS:
        combined_far = np.mean(thresh_fars_a[t] + thresh_fars_b[t])
        combined_frr = np.mean(thresh_frrs_a[t] + thresh_frrs_b[t])
        marker = " ◄" if t == 0.5 else ""
        print(f"  {t:<12.1f} {combined_far:>10.1%} {combined_frr:>10.1%}{marker}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bank-vs-bank continuous auth evaluation")
    parser.add_argument("--user_a", required=True, help="First bank user ID")
    parser.add_argument("--user_b", required=True, help="Second bank user ID")
    parser.add_argument("--runs",   type=int, default=1,
                        help="Number of evaluation runs to average (default: 1)")
    parser.add_argument("--fixed",  action="store_true",
                        help="Use fixed RNG seeds for reproducibility")
    parser.add_argument("--seed",   type=int, default=DEFAULT_SEED,
                        help=f"Base RNG seed (only used with --fixed, default={DEFAULT_SEED})")
    args = parser.parse_args()

    print(f"[Config] user_a={args.user_a}  user_b={args.user_b}  Runs={args.runs}  "
          f"Fixed={'yes (seed=' + str(args.seed) + ')' if args.fixed else 'no'}")

    # Load both models (and their held-out test windows)
    print()
    model_a = load_model(args.user_a)
    model_b = load_model(args.user_b)

    # Check feature cols match — models must have been trained with the same feature set
    if model_a["feature_cols"] != model_b["feature_cols"]:
        print("[Warning] Feature columns differ between the two models. "
              "Results may be unreliable.")

    print(f"\n{'═' * 60}")
    print(f"  {args.user_a} model  vs  {args.user_b} data  (and vice versa)")
    print(f"  {args.runs} run(s), {N_SETS} sets per run, {SET_SIZE} windows per set")
    print(f"{'═' * 60}")

    # Direction A: user_a's model, user_b as impostor
    print(f"\n[Direction 1] Model={args.user_a}  Legitimate={args.user_a}  Impostor={args.user_b}")
    run_results_a = evaluate_direction(
        args.user_a, model_a,
        args.user_b, model_b["test_samples"],
        args.runs, args.seed, args.fixed,
    )

    # Direction B: user_b's model, user_a as impostor
    print(f"\n[Direction 2] Model={args.user_b}  Legitimate={args.user_b}  Impostor={args.user_a}")
    run_results_b = evaluate_direction(
        args.user_b, model_b,
        args.user_a, model_a["test_samples"],
        args.runs, args.seed, args.fixed,
    )

    # Print per-direction summaries
    print(f"\n{'═' * 60}")
    print(f"  Results")
    print(f"{'═' * 60}")

    thresh_fars_a, thresh_frrs_a = print_direction_summary(
        f"Direction 1: {args.user_a} model  |  impostor = {args.user_b}",
        run_results_a, args.runs,
    )
    thresh_fars_b, thresh_frrs_b = print_direction_summary(
        f"Direction 2: {args.user_b} model  |  impostor = {args.user_a}",
        run_results_b, args.runs,
    )

    print_aggregate(thresh_fars_a, thresh_frrs_a, thresh_fars_b, thresh_frrs_b)


if __name__ == "__main__":
    main()