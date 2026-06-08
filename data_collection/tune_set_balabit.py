"""
tune_set_balabit.py

Identical to tune_set_size.py but uses all Balabit users as the impostor
pool instead of a single bank user. For each set size, one impostor user
is randomly chosen per set (matching the approach in eval_continuous_auth.py).

Usage:
    python data_collection/tune_set_balabit.py
    python data_collection/tune_set_balabit.py --runs 50
    python data_collection/tune_set_balabit.py --runs 50 --fixed
    python data_collection/tune_set_balabit.py --runs 50 --min_size 3 --max_size 15
    python data_collection/tune_set_balabit.py --runs 50 --percentile 20
"""

import os
import sys
import argparse
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_SET_SIZE   = 1
MAX_SET_SIZE   = 20
N_SETS         = 20
N_LEGIT        = N_SETS // 2
N_IMPOSTOR     = N_SETS // 2
DEFAULT_RUNS   = 50
DEFAULT_SEED   = 42

CHECKPOINT_DIR = "checkpoints_shen_scroll_bank"
IMPOSTOR_DIR   = "balabit_dataset/training_files"
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

    test_samples = state["test_samples"]
    n_calib      = len(test_samples) // 2

    return {
        "model":         model,
        "scaler":        state["scaler"],
        "reference":     state["reference"],
        "dist_mean":     state["dist_mean"],
        "dist_std":      state["dist_std"],
        "feature_cols":  state["feature_cols"],
        "more_scroll":   state.get("more_scroll", False),
        "dir_scroll":    state.get("dir_scroll",  False),
        "calib_samples": test_samples[:n_calib],   # first half — calibration only
        "eval_samples":  test_samples[n_calib:],   # second half — evaluation only
    }


# ── Data extraction ───────────────────────────────────────────────────────────

def extract_windows_for_user(user_dir, user_id, feature_cols,
                              more_scroll=False, dir_scroll=False):
    from measurements.extract_features_scroll import extract_session_features
    if not os.path.isdir(user_dir):
        return np.empty((0, len(feature_cols)))
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

def calibrate_threshold(m, set_size):
    calib = m["calib_samples"]
    n     = len(calib)

    if n < set_size:
        return 0.5

    x_norm     = preprocess(calib, m)
    raw_scores = m["model"].decision_function(x_norm)
    accepted   = (raw_scores >= 0).astype(int)

    acceptance_rates = []
    for start in range(n - set_size + 1):
        block = accepted[start:start + set_size]
        acceptance_rates.append(block.mean())

    step      = 1.0 / set_size
    min_rate  = min(acceptance_rates)
    threshold = max(0.0, min(1.0, min_rate - step))
    threshold = round(round(threshold / step) * step, 10)
    return threshold


# ── Set generation ────────────────────────────────────────────────────────────

def sample_contiguous_block(windows, set_size, rng):
    max_start = len(windows) - set_size
    start     = int(rng.integers(0, max_start + 1))
    return windows[start:start + set_size], start


def generate_sets(legit_windows, impostor_pool, set_size, rng):
    """
    impostor_pool: dict {user_id: np.ndarray}
    One impostor user is randomly chosen per impostor set.
    """
    impostor_users = [u for u, w in impostor_pool.items() if len(w) >= set_size]
    if not impostor_users:
        return None

    sets = []
    for i in range(N_LEGIT):
        block, _ = sample_contiguous_block(legit_windows, set_size, rng)
        sets.append({"label": "legitimate", "windows": block})
    for i in range(N_IMPOSTOR):
        imp_user = impostor_users[int(rng.integers(0, len(impostor_users)))]
        block, _ = sample_contiguous_block(impostor_pool[imp_user], set_size, rng)
        sets.append({"label": "impostor", "windows": block})
    return sets


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_set(set_dict, m, threshold):
    x_norm          = preprocess(set_dict["windows"], m)
    raw_scores      = m["model"].decision_function(x_norm)
    sample_accepted = [float(s) >= 0 for s in raw_scores]
    accepted_rate   = sum(sample_accepted) / len(sample_accepted)
    return accepted_rate >= threshold


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_set_size(m, impostor_pool, set_size, threshold, runs, base_seed, fixed):
    legit_windows = m["eval_samples"]

    if len(legit_windows) < set_size:
        return None

    legit_accepted_per_run  = []
    impost_rejected_per_run = []

    for run_idx in range(runs):
        seed = (base_seed + run_idx) if fixed else None
        rng  = np.random.default_rng(seed)

        sets = generate_sets(legit_windows, impostor_pool, set_size, rng)
        if sets is None:
            return None

        legit_acc  = 0
        impost_rej = 0
        for s in sets:
            verdict = score_set(s, m, threshold)
            if s["label"] == "legitimate" and verdict:
                legit_acc  += 1
            elif s["label"] == "impostor" and not verdict:
                impost_rej += 1

        legit_accepted_per_run.append(legit_acc)
        impost_rejected_per_run.append(impost_rej)

    return legit_accepted_per_run, impost_rejected_per_run


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_table(rows):
    header = (f"{'Set Size':>8}  {'Threshold':>9}  "
              f"{'FAR mean':>9}  {'FAR std':>7}  "
              f"{'FRR mean':>9}  {'FRR std':>7}  "
              f"{'Legit Acc':>12}  {'Impost Rej':>12}")
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))

    for r in rows:
        print(f"  {r['set_size']:>6}  {r['threshold']:>9.1f}  "
              f"{r['far_mean']:>8.1%}  {r['far_std']:>7.1%}  "
              f"{r['frr_mean']:>8.1%}  {r['frr_std']:>7.1%}  "
              f"  {r['legit_mean']:.1f}/{N_LEGIT:>3}  "
              f"  {r['impost_mean']:.1f}/{N_IMPOSTOR:>3}")

    print("─" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find minimum set size using Balabit impostors")
    parser.add_argument("--runs",       type=int,   default=DEFAULT_RUNS)
    parser.add_argument("--fixed",      action="store_true")
    parser.add_argument("--seed",       type=int,   default=DEFAULT_SEED)
    parser.add_argument("--min_size",   type=int,   default=MIN_SET_SIZE)
    parser.add_argument("--max_size",   type=int,   default=MAX_SET_SIZE)
    args = parser.parse_args()

    user_a = input("Enter model user ID (legitimate): ").strip()

    print(f"\n[Config] Model={user_a}  Impostor=Balabit  Runs={args.runs}  "
          f"Set sizes={args.min_size}-{args.max_size}  "
          f"Fixed={'yes (seed=' + str(args.seed) + ')' if args.fixed else 'no'}")

    print()
    model_a = load_model(user_a)

    # Load all Balabit impostor users
    print(f"[Data] Loading Balabit impostor pool from {IMPOSTOR_DIR} ...")
    impostor_pool = {}
    for imp_user in sorted(os.listdir(IMPOSTOR_DIR)):
        imp_dir  = os.path.join(IMPOSTOR_DIR, imp_user)
        if not os.path.isdir(imp_dir):
            continue
        windows = extract_windows_for_user(
            imp_dir, imp_user,
            model_a["feature_cols"],
            model_a["more_scroll"],
            model_a["dir_scroll"],
        )
        if len(windows) >= 1:
            impostor_pool[imp_user] = windows
    print(f"[Data] {len(impostor_pool)} Balabit users loaded")
    print(f"[Data] Legitimate calibration: {len(model_a['calib_samples'])} windows")
    print(f"[Data] Legitimate evaluation:  {len(model_a['eval_samples'])} windows")

    rows = []
    print(f"\n[Sweep] Testing set sizes {args.min_size} to {args.max_size} ...")

    for set_size in range(args.min_size, args.max_size + 1):
        threshold = calibrate_threshold(model_a, set_size)

        result = evaluate_set_size(
            model_a, impostor_pool, set_size, threshold,
            args.runs, args.seed, args.fixed
        )

        if result is None:
            print(f"  [Skip] Set size {set_size}: not enough windows.")
            continue

        legit_accepted_runs, impost_rejected_runs = result

        far_per_run = [(N_IMPOSTOR - r) / N_IMPOSTOR for r in impost_rejected_runs]
        frr_per_run = [(N_LEGIT    - r) / N_LEGIT    for r in legit_accepted_runs]

        rows.append({
            "set_size":    set_size,
            "threshold":   threshold,
            "far_mean":    np.mean(far_per_run),
            "far_std":     np.std(far_per_run),
            "frr_mean":    np.mean(frr_per_run),
            "frr_std":     np.std(frr_per_run),
            "legit_mean":  np.mean(legit_accepted_runs),
            "impost_mean": np.mean(impost_rejected_runs),
        })

        print(f"  Set size {set_size:>2}: T={threshold:.1f}  "
              f"FAR={np.mean(far_per_run):.1%}  FRR={np.mean(frr_per_run):.1%}  "
              f"Legit={np.mean(legit_accepted_runs):.1f}/{N_LEGIT}  "
              f"Impost={np.mean(impost_rejected_runs):.1f}/{N_IMPOSTOR}")

    print_table(rows)


if __name__ == "__main__":
    main()