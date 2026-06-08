"""
bank_to_bank_1d_tune.py

Identical to bank_to_bank_1d.py but derives the consensus threshold
automatically from the legitimate user's held-out data, without requiring
any impostor data — making it suitable for real deployment.

Threshold calibration (inspired by Mondal & Bours, 2013):
  After training, the held-out legitimate windows are scored in contiguous
  blocks of SET_SIZE. The acceptance rate of each block is recorded. The
  consensus threshold is set equal to the minimum observed acceptance rate.
  This prioritises low FAR over low FRR — the legitimate user may occasionally
  be rejected on a bad block, but impostors must clear the same bar the
  legitimate user themselves set during calibration.

  Example: if the minimum observed acceptance rate across all legitimate
  blocks is 8/10 = 0.8, the threshold is set to 0.8.

Usage:
    python data_collection/bank_to_bank_1d_tune.py
    python data_collection/bank_to_bank_1d_tune.py --runs 50
    python data_collection/bank_to_bank_1d_tune.py --runs 50 --fixed
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
BANK_DATA_DIR  = "bank_collection/bank-data"
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

    n_calib = len(state["test_samples"]) // 2
    print(f"[Model] Loaded for '{user_id}' — "
          f"train={state['n_train']} windows, held-out={state['n_test']} windows "
          f"({n_calib} calibration, {state['n_test'] - n_calib} evaluation), "
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
        "test_samples": state["test_samples"],          # full held-out set
        "calib_samples": state["test_samples"][:n_calib],   # first half — calibration only
        "eval_samples":  state["test_samples"][n_calib:],   # second half — evaluation only
    }


# ── Data extraction ───────────────────────────────────────────────────────────

def extract_windows_for_user(user_id, feature_cols, more_scroll=False, dir_scroll=False):
    from measurements.extract_features_scroll import extract_session_features
    user_dir = os.path.join(BANK_DATA_DIR, user_id)
    if not os.path.isdir(user_dir):
        print(f"[Error] No data directory found at {user_dir}")
        sys.exit(1)
    session_files = sorted([
        os.path.join(user_dir, f) for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])
    all_vecs = []
    for path in session_files:
        try:
            df = extract_session_features(path, user_id, window_size=WINDOW_SIZE,
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
    return np.array(all_vecs) if all_vecs else np.empty((0, len(feature_cols)))


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(windows, m):
    x = m["scaler"].transform(np.array(windows))
    x = np.abs(x - m["reference"])
    x = (x - m["dist_mean"]) / np.where(m["dist_std"] < 1e-9, 1.0, m["dist_std"])
    return x


# ── Threshold calibration ─────────────────────────────────────────────────────

def calibrate_threshold(m):
    """
    Derive the consensus threshold from the first half of the legitimate
    user's held-out windows, without requiring any impostor data.

    The held-out windows are split in two:
      - First half  → calibration (used here to set the threshold)
      - Second half → evaluation  (used in evaluate() to score legitimate sets)

    Score the calibration windows in all possible contiguous blocks of SET_SIZE.
    Record the acceptance rate of each block. Set the consensus threshold to
    one discrete step below the minimum observed acceptance rate, ensuring the
    legitimate user is never locked out.

    This mirrors the person-specific threshold approach of Mondal & Bours (2013),
    applied to a one-class consensus system rather than a two-class trust model.
    """
    all_legit  = m["calib_samples"]
    n_calib    = len(all_legit)

    if n_calib < SET_SIZE:
        print(f"[Warning] Only {n_calib} calibration windows — fewer than SET_SIZE={SET_SIZE}. "
              f"Defaulting threshold to 0.5.")
        return 0.5

    x_norm     = preprocess(all_legit, m)
    raw_scores = m["model"].decision_function(x_norm)
    accepted   = (raw_scores >= 0).astype(int)

    acceptance_rates = []
    for start in range(n_calib - SET_SIZE + 1):
        block = accepted[start:start + SET_SIZE]
        acceptance_rates.append(block.mean())

    min_rate  = min(acceptance_rates)
    mean_rate = np.mean(acceptance_rates)
    step      = 1.0 / SET_SIZE

    tuned_threshold = max(0.0, min(1.0, min_rate))
    tuned_threshold = round(round(tuned_threshold / step) * step, 10)

    print(f"\n[Calibration] {n_calib} calibration windows, "
          f"{n_calib - SET_SIZE + 1} blocks of {SET_SIZE}:")
    print(f"  Min acceptance rate  : {min_rate:.1%}  "
          f"({int(min_rate * SET_SIZE)}/{SET_SIZE} windows)")
    print(f"  Mean acceptance rate : {mean_rate:.1%}")
    print(f"  Tuned consensus threshold: {tuned_threshold:.1f} "
          f"(equal to min = {min_rate:.1%})")

    return tuned_threshold


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
    x_norm          = preprocess(set_dict["windows"], m)
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

def evaluate(target_user, target_model, impostor_user, impostor_windows,
             tuned_threshold, runs, base_seed, fixed):
    legit_windows = target_model["eval_samples"]

    if len(legit_windows) < SET_SIZE:
        print(f"[Error] {target_user} has only {len(legit_windows)} held-out windows "
              f"(need {SET_SIZE}).")
        sys.exit(1)
    if len(impostor_windows) < SET_SIZE:
        print(f"[Error] {impostor_user} has only {len(impostor_windows)} windows "
              f"(need {SET_SIZE}).")
        sys.exit(1)

    all_run_results = []
    for run_idx in range(runs):
        seed = (base_seed + run_idx) if fixed else None
        rng  = np.random.default_rng(seed)

        sets = generate_sets(target_user, legit_windows, impostor_user, impostor_windows, rng)

        results = []
        for s in sets:
            raw_scores, sample_accepted = score_set(s, target_model)
            sample_acc       = per_sample_accuracy(sample_accepted, s["label"])
            # Majority vote uses the tuned threshold
            majority_verdict = consensus_verdict(sample_accepted, threshold=tuned_threshold)
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

def print_results(all_run_results, runs, user_a, user_b, tuned_threshold):
    sample_fars, sample_frrs = [], []
    thresh_fars = {t: [] for t in THRESHOLDS}
    thresh_frrs = {t: [] for t in THRESHOLDS}

    print(f"\n{'─' * 60}")
    print(f"  Per-run verdicts at tuned threshold (T={tuned_threshold:.1f})")
    print(f"{'─' * 60}")

    for run_idx, results in enumerate(all_run_results):
        legit_results    = [r for r in results if r["label"] == "legitimate"]
        impostor_results = [r for r in results if r["label"] == "impostor"]

        legit_accepted  = sum(r["consensus"][tuned_threshold] for r in legit_results)
        impost_rejected = sum(not r["consensus"][tuned_threshold] for r in impostor_results)

        # Overall session verdict: impostor detected if majority of impostor sets rejected
        impostor_detected = impost_rejected > len(impostor_results) / 2
        verdict = "IMPOSTER REJECTED" if impostor_detected else "IMPOSTER ACCEPTED"

        print(f"  Run {run_idx + 1:>2}: "
              f"Legitimate accepted: {legit_accepted}/{len(legit_results)},  "
              f"Impostors rejected: {impost_rejected}/{len(impostor_results)}  "
              f"→ {verdict}")

        sf, sr = compute_sample_far_frr(results)
        sample_fars.append(sf)
        sample_frrs.append(sr)
        for t in THRESHOLDS:
            far, frr = compute_metrics(results, t)
            thresh_fars[t].append(far)
            thresh_frrs[t].append(frr)

    print(f"\n{'═' * 60}")
    print(f"  Model={user_a} (OCSVM)  Legitimate={user_a}  Impostor={user_b}")
    print(f"  {runs} run(s), {N_SETS} sets per run, {SET_SIZE} windows per set")
    print(f"  Tuned consensus threshold: {tuned_threshold:.1f}")
    print(f"{'═' * 60}")

    print(f"\nPer-window:  FAR={np.mean(sample_fars):.1%} (std {np.std(sample_fars):.1%})  "
          f"FRR={np.mean(sample_frrs):.1%} (std {np.std(sample_frrs):.1%})")

    if runs > 1:
        print(f"\n  {'Threshold':<12} {'FAR mean':>10} {'FAR std':>9} {'FRR mean':>10} {'FRR std':>9}")
    else:
        print(f"\n  {'Threshold':<12} {'FAR':>8} {'FRR':>8}")

    for t in THRESHOLDS:
        # Mark the tuned threshold with ◄
        marker = " ◄ (tuned)" if abs(t - tuned_threshold) < 1e-9 else ""
        if runs > 1:
            print(f"  {t:<12.1f} "
                  f"{np.mean(thresh_fars[t]):>10.1%} {np.std(thresh_fars[t]):>9.1%} "
                  f"{np.mean(thresh_frrs[t]):>10.1%} {np.std(thresh_frrs[t]):>9.1%}{marker}")
        else:
            print(f"  {t:<12.1f} {thresh_fars[t][0]:>8.1%} {thresh_frrs[t][0]:>8.1%}{marker}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bank-vs-bank evaluation with calibrated consensus threshold")
    parser.add_argument("--runs",  type=int, default=1,
                        help="Number of evaluation runs to average (default: 1)")
    parser.add_argument("--fixed", action="store_true",
                        help="Use fixed RNG seeds for reproducibility")
    parser.add_argument("--seed",  type=int, default=DEFAULT_SEED,
                        help=f"Base RNG seed (only used with --fixed, default={DEFAULT_SEED})")
    args = parser.parse_args()

    user_a = input("Enter model user ID (legitimate): ").strip()
    user_b = input("Enter impostor user ID: ").strip()

    print(f"\n[Config] Model={user_a}  Impostor={user_b}  Runs={args.runs}  "
          f"Fixed={'yes (seed=' + str(args.seed) + ')' if args.fixed else 'no'}")

    print()
    model_a = load_model(user_a)

    # Calibrate threshold from legitimate held-out data only
    tuned_threshold = calibrate_threshold(model_a)

    print(f"\n[Data] Extracting windows for impostor '{user_b}' from {BANK_DATA_DIR}/{user_b} ...")
    impostor_windows = extract_windows_for_user(
        user_b, model_a["feature_cols"],
        model_a["more_scroll"], model_a["dir_scroll"]
    )
    if len(impostor_windows) < SET_SIZE:
        print(f"[Error] Not enough windows for impostor '{user_b}' "
              f"({len(impostor_windows)} found, need {SET_SIZE}).")
        sys.exit(1)
    print(f"[Data] Impostor '{user_b}': {len(impostor_windows)} windows")

    all_run_results = evaluate(
        user_a, model_a,
        user_b, impostor_windows,
        tuned_threshold,
        args.runs, args.seed, args.fixed,
    )

    print_results(all_run_results, args.runs, user_a, user_b, tuned_threshold)


if __name__ == "__main__":
    main()