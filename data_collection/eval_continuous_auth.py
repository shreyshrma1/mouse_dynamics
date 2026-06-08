"""
eval_continuous_auth.py

Preliminary evaluation of the continuous authentication system using
contiguous windows of mouse data.

Legitimate data: bank_collection/bank-data/<user_id>/  (raw session CSVs)
Impostor data:   balabit_dataset/training_files/<imp_user>/  (raw session CSVs)

Both are processed through extract_session_features → StandardScaler →
Shen distance transform → OCSVM, matching the pipeline in
train_from_existing_shen_scroll.py and evaluate_shen_scroll.py.

For a specified target user, generates N_SETS pure sets of SET_SIZE contiguous windows:
  - N_LEGIT    sets: all windows from the target user
  - N_IMPOSTOR sets: all windows from a single randomly chosen impostor

For each set the model scores all windows, then:
  - Per-sample accuracy is computed (correct accept/reject decisions)
  - Consensus verdict is computed under majority vote and a threshold sweep
  - FAR and FRR are reported at each threshold

Robustness flags:
  --runs N     Repeat the full evaluation N times with different seeds and
               average the metrics. Reduces variance from a single lucky/unlucky
               sampling draw. (default: 1)
  --all_users  Run evaluation for every user found in bank-data/ and report
               per-user and aggregate metrics.

Usage:
    python eval_continuous_auth.py --user shrey
    python eval_continuous_auth.py --user shrey --fixed
    python eval_continuous_auth.py --user shrey --fixed --seed 99
    python eval_continuous_auth.py --user shrey --runs 10
    python eval_continuous_auth.py --all_users --runs 5
"""

import os
import sys
import argparse
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants ─────────────────────────────────────────────────────────────────

N_SETS        = 20
SET_SIZE      = 5
N_LEGIT       = N_SETS // 2
N_IMPOSTOR    = N_SETS // 2
THRESHOLDS    = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_SEED  = 42

BANK_DATA_DIR  = "bank_collection/bank-data"
IMPOSTOR_DIR   = "balabit_dataset/training_files"
CHECKPOINT_DIR = "checkpoints_shen_scroll_bank"
WINDOW_SIZE    = 5


# ── Data loading ──────────────────────────────────────────────────────────────

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
    Extract all windows from a user's session files, preserving row order
    so contiguous slices are temporally meaningful.
    Returns np.ndarray of shape (n_windows, n_features).
    """
    from measurements.extract_features_scroll import extract_session_features

    session_files = get_session_files(user_dir)
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
          f"features={len(state['feature_cols'])}, "
          f"nu={state['nu']}, gamma={state['gamma']}")

    return (
        model,
        state["scaler"],
        state["reference"],
        state["dist_mean"],
        state["dist_std"],
        state["feature_cols"],
        state.get("more_scroll", False),
        state.get("dir_scroll",  False),
        state["test_samples"],   # held-out windows, never seen during training
    )


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(windows, scaler, reference, dist_mean, dist_std):
    """StandardScaler → Shen distance → normalize, matching evaluate_shen_scroll.py."""
    x = scaler.transform(np.array(windows))
    x = np.abs(x - reference)
    x = (x - dist_mean) / np.where(dist_std < 1e-9, 1.0, dist_std)
    return x


# ── Set generation ────────────────────────────────────────────────────────────

def sample_contiguous_block(windows, rng):
    max_start = len(windows) - SET_SIZE
    start = int(rng.integers(0, max_start + 1))
    return windows[start:start + SET_SIZE], start


def generate_sets(target_user, legit_windows, impostor_pool, rng):
    """
    impostor_pool: dict {user_id: np.ndarray} of all usable impostor users.
    Returns list of set dicts.
    """
    impostor_users = list(impostor_pool.keys())
    if not impostor_users:
        print("[Error] No impostor users with enough windows found.")
        sys.exit(1)

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
        imp_user = impostor_users[int(rng.integers(0, len(impostor_users)))]
        block, start = sample_contiguous_block(impostor_pool[imp_user], rng)
        sets.append({
            "set_id":  N_LEGIT + i + 1,
            "label":   "impostor",
            "user":    imp_user,
            "start":   start,
            "windows": block,
        })

    return sets


# ── Scoring & consensus ───────────────────────────────────────────────────────

def score_set(set_dict, model, scaler, reference, dist_mean, dist_std):
    x_norm = preprocess(set_dict["windows"], scaler, reference, dist_mean, dist_std)
    raw_scores      = model.decision_function(x_norm)
    sample_accepted = [float(s) >= 0 for s in raw_scores]
    return raw_scores, sample_accepted


def consensus_verdict(sample_accepted, threshold):
    return (sum(sample_accepted) / len(sample_accepted)) >= threshold


def per_sample_accuracy(sample_accepted, label):
    if label == "legitimate":
        correct = sum(sample_accepted)
    else:
        correct = sum(not a for a in sample_accepted)
    return correct / len(sample_accepted)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results, threshold):
    """
    FAR  = impostor sets incorrectly accepted / total impostor sets
    FRR  = legitimate sets incorrectly rejected / total legitimate sets
    Both are reported as fractions (multiply by 100 for %).
    """
    legit_results    = [r for r in results if r["label"] == "legitimate"]
    impostor_results = [r for r in results if r["label"] == "impostor"]

    # A legitimate set is incorrectly rejected if consensus verdict is reject
    frr = sum(
        not r["consensus"][threshold] for r in legit_results
    ) / len(legit_results) if legit_results else 0.0

    # An impostor set is incorrectly accepted if consensus verdict is accept
    far = sum(
        r["consensus"][threshold] for r in impostor_results
    ) / len(impostor_results) if impostor_results else 0.0

    return far, frr


def compute_sample_far_frr(results):
    """
    FAR and FRR at the individual window level (before consensus),
    averaged across all sets of the relevant type.
    """
    legit_results    = [r for r in results if r["label"] == "legitimate"]
    impostor_results = [r for r in results if r["label"] == "impostor"]

    # Per-window FRR: fraction of legitimate windows rejected
    sample_frr = np.mean([
        sum(not a for a in r["sample_accepted"]) / len(r["sample_accepted"])
        for r in legit_results
    ]) if legit_results else 0.0

    # Per-window FAR: fraction of impostor windows accepted
    sample_far = np.mean([
        sum(r["sample_accepted"]) / len(r["sample_accepted"])
        for r in impostor_results
    ]) if impostor_results else 0.0

    return sample_far, sample_frr


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(results, runs=1):
    """
    Print per-set table and summary. If runs > 1, results is a list of
    run-result lists and metrics are averaged across runs.
    """
    # For multi-run mode, print summary only (per-set table doesn't aggregate cleanly)
    if runs == 1:
        _print_per_set_table(results[0])
        _print_summary(results[0])
    else:
        print(f"\n[Multi-run] Averaging metrics over {runs} runs ({N_SETS} sets each)\n")
        _print_averaged_summary(results, runs)


def _print_per_set_table(results):
    thresh_headers = "".join(f"  T={t:.1f}" for t in THRESHOLDS)
    header = (f"{'Set':>4}  {'Type':<12}  {'User':<12}  {'Start':>5}  "
              f"{'Sample Acc':>10}  {'Maj Vote':>9}{thresh_headers}")
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))

    for r in results:
        thresh_verdicts = "".join(
            f"  {'ACC' if r['consensus'][t] else 'REJ':>5}"
            for t in THRESHOLDS
        )
        maj_correct = "✓" if r["majority_correct"] else "✗"
        print(
            f"{r['set_id']:>4}  {r['label']:<12}  {r['user']:<12}  "
            f"{r['start']:>5}  {r['sample_acc']:>10.1%}  "
            f"{'ACC' if r['majority_verdict'] else 'REJ':>6} {maj_correct:>2}"
            f"{thresh_verdicts}"
        )
    print("─" * len(header))


def _print_summary(results):
    legit_results    = [r for r in results if r["label"] == "legitimate"]
    impostor_results = [r for r in results if r["label"] == "impostor"]

    sample_far, sample_frr = compute_sample_far_frr(results)

    print(f"\n{'── Summary ':─<60}")
    print(f"\nPer-window (before consensus):")
    print(f"  FAR : {sample_far:.1%}  (impostor windows incorrectly accepted)")
    print(f"  FRR : {sample_frr:.1%}  (legitimate windows incorrectly rejected)")

    print(f"\nPer-set sample accuracy:")
    print(f"  Legitimate sets : {np.mean([r['sample_acc'] for r in legit_results]):.1%}  "
          f"(std {np.std([r['sample_acc'] for r in legit_results]):.1%})")
    print(f"  Impostor sets   : {np.mean([r['sample_acc'] for r in impostor_results]):.1%}  "
          f"(std {np.std([r['sample_acc'] for r in impostor_results]):.1%})")

    print(f"\nConsensus FAR / FRR by threshold ({N_SETS} sets):")
    print(f"  {'Threshold':<12} {'FAR':>8} {'FRR':>8}")
    for t in THRESHOLDS:
        far, frr = compute_metrics(results, t)
        marker = " ◄" if t == 0.5 else ""
        print(f"  {t:<12.1f} {far:>8.1%} {frr:>8.1%}{marker}")

    # Majority vote (T=0.5) callout
    far_maj, frr_maj = compute_metrics(results, 0.5)
    print(f"\nMajority vote (T=0.5):  FAR={far_maj:.1%}  FRR={frr_maj:.1%}\n")


def _print_averaged_summary(all_run_results, runs):
    """Average FAR/FRR across multiple runs."""
    sample_fars, sample_frrs = [], []
    thresh_fars  = {t: [] for t in THRESHOLDS}
    thresh_frrs  = {t: [] for t in THRESHOLDS}

    for results in all_run_results:
        sf, sr = compute_sample_far_frr(results)
        sample_fars.append(sf)
        sample_frrs.append(sr)
        for t in THRESHOLDS:
            far, frr = compute_metrics(results, t)
            thresh_fars[t].append(far)
            thresh_frrs[t].append(frr)

    print(f"Per-window (before consensus):")
    print(f"  FAR : {np.mean(sample_fars):.1%}  (std {np.std(sample_fars):.1%})")
    print(f"  FRR : {np.mean(sample_frrs):.1%}  (std {np.std(sample_frrs):.1%})")

    print(f"\nConsensus FAR / FRR by threshold (mean ± std over {runs} runs):")
    print(f"  {'Threshold':<12} {'FAR mean':>10} {'FAR std':>9} {'FRR mean':>10} {'FRR std':>9}")
    for t in THRESHOLDS:
        marker = " ◄" if t == 0.5 else ""
        print(f"  {t:<12.1f} "
              f"{np.mean(thresh_fars[t]):>10.1%} {np.std(thresh_fars[t]):>9.1%} "
              f"{np.mean(thresh_frrs[t]):>10.1%} {np.std(thresh_frrs[t]):>9.1%}{marker}")


# ── Single-user evaluation ────────────────────────────────────────────────────

def evaluate_user(user_id, legit_windows, impostor_pool, model_artifacts, runs, base_seed, fixed):
    """
    Run the full evaluation for one user. Returns list of run-result lists.
    model_artifacts: (model, scaler, reference, dist_mean, dist_std)
    """
    model, scaler, reference, dist_mean, dist_std = model_artifacts
    all_run_results = []

    for run_idx in range(runs):
        if fixed:
            seed = base_seed + run_idx
        else:
            seed = None
        rng = np.random.default_rng(seed)

        sets = generate_sets(user_id, legit_windows, impostor_pool, rng)

        results = []
        for s in sets:
            raw_scores, sample_accepted = score_set(
                s, model, scaler, reference, dist_mean, dist_std
            )
            sample_acc       = per_sample_accuracy(sample_accepted, s["label"])
            majority_verdict = consensus_verdict(sample_accepted, threshold=0.5)
            majority_correct = (majority_verdict == (s["label"] == "legitimate"))

            consensus         = {t: consensus_verdict(sample_accepted, t) for t in THRESHOLDS}
            consensus_correct = {
                t: (consensus[t] == (s["label"] == "legitimate"))
                for t in THRESHOLDS
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Continuous auth evaluation")
    parser.add_argument("--user",      default=None,
                        help="Target user ID (must exist in bank-data/). "
                             "Omit if using --all_users.")
    parser.add_argument("--all_users", action="store_true",
                        help="Evaluate every user found in bank-data/ and report aggregate metrics.")
    parser.add_argument("--fixed",     action="store_true",
                        help="Use a fixed RNG seed for reproducible sets.")
    parser.add_argument("--seed",      type=int, default=DEFAULT_SEED,
                        help=f"Base RNG seed (only used with --fixed, default={DEFAULT_SEED}). "
                             f"For multi-run, each run uses seed+run_index.")
    parser.add_argument("--runs",      type=int, default=1,
                        help="Repeat evaluation N times and average metrics. "
                             "Reduces variance from a single sampling draw. (default: 1)")
    args = parser.parse_args()

    if not args.user and not args.all_users:
        print("[Error] Provide --user <id> or --all_users.")
        sys.exit(1)

    # Determine which users to evaluate
    if args.all_users:
        users = sorted([
            d for d in os.listdir(BANK_DATA_DIR)
            if os.path.isdir(os.path.join(BANK_DATA_DIR, d))
        ])
        print(f"[Config] All-users mode: {len(users)} users found in {BANK_DATA_DIR}")
    else:
        users = [args.user]
        print(f"[Config] User={args.user}  Runs={args.runs}  "
              f"Fixed={'yes (seed=' + str(args.seed) + ')' if args.fixed else 'no'}")

    # Pre-load impostor pool (shared across all target users)
    print(f"[Data] Loading impostor data from {IMPOSTOR_DIR} ...")
    # Feature cols determined per-user from their model; we load impostors lazily per user
    # to ensure the correct feature set is used. Store raw dirs here.
    impostor_user_dirs = {
        u: os.path.join(IMPOSTOR_DIR, u)
        for u in sorted(os.listdir(IMPOSTOR_DIR))
        if os.path.isdir(os.path.join(IMPOSTOR_DIR, u))
    }
    print(f"[Data] {len(impostor_user_dirs)} impostor user directories found.")

    # Per-user evaluation
    aggregate_fars = {t: [] for t in THRESHOLDS}
    aggregate_frrs = {t: [] for t in THRESHOLDS}

    for user_id in users:
        print(f"\n{'═' * 60}")
        print(f"  Evaluating: {user_id}")
        print(f"{'═' * 60}")

        # Load model and held-out test windows for this user
        (model, scaler, reference, dist_mean,
         dist_std, feature_cols, more_scroll, dir_scroll,
         test_samples) = load_model(user_id)

        # test_samples are saved post-split but pre-scaling in train_from_existing
        # (the scaler is only fit+applied to train_samples before model fitting).
        # They are therefore already in raw feature space — preprocess() will
        # apply the full pipeline (scaler -> distance -> normalize) correctly.
        legit_windows = test_samples

        if len(legit_windows) < SET_SIZE:
            print(f"[Skip] Not enough held-out windows ({len(legit_windows)}) for {user_id}.")
            continue
        print(f"[Data] Legitimate (held-out only): {len(legit_windows)} windows")

        # Load impostor windows using this user's feature cols
        impostor_pool = {}
        for imp_user, imp_dir in impostor_user_dirs.items():
            imp_windows = extract_windows_for_user(
                imp_dir, imp_user, feature_cols, more_scroll, dir_scroll
            )
            if len(imp_windows) >= SET_SIZE:
                impostor_pool[imp_user] = imp_windows
        print(f"[Data] Impostor pool: {len(impostor_pool)} usable users")

        model_artifacts = (model, scaler, reference, dist_mean, dist_std)
        all_run_results = evaluate_user(
            user_id, legit_windows, impostor_pool,
            model_artifacts, args.runs, args.seed, args.fixed
        )

        print_results(all_run_results, runs=args.runs)

        # Accumulate for aggregate report
        if args.all_users:
            for results in all_run_results:
                for t in THRESHOLDS:
                    far, frr = compute_metrics(results, t)
                    aggregate_fars[t].append(far)
                    aggregate_frrs[t].append(frr)

    # Aggregate report across all users
    if args.all_users and len(users) > 1:
        print(f"\n{'═' * 60}")
        print(f"  Aggregate across all users ({len(users)} users, {args.runs} run(s) each)")
        print(f"{'═' * 60}")
        print(f"\n  {'Threshold':<12} {'FAR mean':>10} {'FAR std':>9} {'FRR mean':>10} {'FRR std':>9}")
        for t in THRESHOLDS:
            marker = " ◄" if t == 0.5 else ""
            print(f"  {t:<12.1f} "
                  f"{np.mean(aggregate_fars[t]):>10.1%} {np.std(aggregate_fars[t]):>9.1%} "
                  f"{np.mean(aggregate_frrs[t]):>10.1%} {np.std(aggregate_frrs[t]):>9.1%}{marker}")
        print()


if __name__ == "__main__":
    main()