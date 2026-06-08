"""
auth_sim.py

Simulates the deployment authentication decision for a single user pair.

For each run:
  - A random contiguous block of SET_SIZE windows is drawn from the
    legitimate user's eval pool (simulating a real login session)
  - A random contiguous block of SET_SIZE windows is drawn from the
    impostor user's data
  - One binary decision is made for each: accepted or rejected
  - The decision is compared against ground truth and reported

This is honest to the deployment flow where a single block of windows
collected at login time determines whether the user is authenticated.

The calibrated threshold is derived from the legitimate user's calibration
pool (first half of held-out windows) using the same method as
bank_to_bank_1d_tune.py — one discrete step below the minimum observed
acceptance rate across contiguous blocks of SET_SIZE.

Usage:
    python data_collection/auth_sim.py
    python data_collection/auth_sim.py --runs 50
    python data_collection/auth_sim.py --runs 50 --set_size 8
    python data_collection/auth_sim.py --runs 50 --fixed
"""

import os
import sys
import argparse
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SET_SIZE = 10
DEFAULT_RUNS     = 50
DEFAULT_SEED     = 42

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
        "calib_samples": test_samples[:n_calib],
        "eval_samples":  test_samples[n_calib:],
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

def calibrate_threshold(m, set_size):
    calib = m["calib_samples"]
    n     = len(calib)

    if n < set_size:
        print(f"[Warning] Only {n} calibration windows, fewer than set_size={set_size}. "
              f"Defaulting to 0.5.")
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


# ── Single authentication decision ───────────────────────────────────────────

def authenticate(windows, m, threshold):
    """
    Make a single binary authentication decision for one block of windows.
    Returns True (accepted) or False (rejected).
    """
    x_norm        = preprocess(windows, m)
    raw_scores    = m["model"].decision_function(x_norm)
    accepted_rate = np.mean(raw_scores >= 0)
    return accepted_rate >= threshold


def sample_block(windows, set_size, rng):
    max_start = len(windows) - set_size
    start     = int(rng.integers(0, max_start + 1))
    return windows[start:start + set_size]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simulate single-block deployment authentication")
    parser.add_argument("--runs",     type=int, default=DEFAULT_RUNS,
                        help=f"Number of simulated login attempts (default: {DEFAULT_RUNS})")
    parser.add_argument("--set_size", type=int, default=DEFAULT_SET_SIZE,
                        help=f"Number of windows per authentication decision "
                             f"(default: {DEFAULT_SET_SIZE})")
    parser.add_argument("--fixed",    action="store_true",
                        help="Use fixed RNG seeds for reproducibility")
    parser.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    user_a = input("Enter model user ID (legitimate): ").strip()
    user_b = input("Enter impostor user ID: ").strip()

    print(f"\n[Config] Model={user_a}  Impostor={user_b}  "
          f"Runs={args.runs}  Set size={args.set_size}  "
          f"Fixed={'yes (seed=' + str(args.seed) + ')' if args.fixed else 'no'}")

    print()
    model_a = load_model(user_a)

    threshold = calibrate_threshold(model_a, args.set_size)
    print(f"[Calibration] Threshold={threshold:.1f} "
          f"(from {len(model_a['calib_samples'])} calibration windows)")

    print(f"[Data] Extracting windows for impostor '{user_b}' ...")
    impostor_windows = extract_windows_for_user(
        user_b, model_a["feature_cols"],
        model_a["more_scroll"], model_a["dir_scroll"]
    )

    legit_windows = model_a["eval_samples"]

    if len(legit_windows) < args.set_size:
        print(f"[Error] Only {len(legit_windows)} eval windows — need at least {args.set_size}.")
        sys.exit(1)
    if len(impostor_windows) < args.set_size:
        print(f"[Error] Only {len(impostor_windows)} impostor windows — need at least {args.set_size}.")
        sys.exit(1)

    print(f"[Data] Legitimate eval: {len(legit_windows)} windows")
    print(f"[Data] Impostor '{user_b}': {len(impostor_windows)} windows")

    legit_correct   = 0
    impost_correct  = 0

    print(f"\n{'─' * 60}")
    print(f"  Simulated login attempts (set size = {args.set_size}, T = {threshold:.1f})")
    print(f"{'─' * 60}")

    for run_idx in range(args.runs):
        seed = (args.seed + run_idx) if args.fixed else None
        rng  = np.random.default_rng(seed)

        legit_block   = sample_block(legit_windows,    args.set_size, rng)
        impost_block  = sample_block(impostor_windows, args.set_size, rng)

        legit_verdict  = authenticate(legit_block,  model_a, threshold)
        impost_verdict = authenticate(impost_block, model_a, threshold)

        legit_ok  = legit_verdict   # correct if accepted
        impost_ok = not impost_verdict  # correct if rejected

        if legit_ok:
            legit_correct  += 1
        if impost_ok:
            impost_correct += 1

        legit_str  = "ACCEPTED ✓" if legit_ok  else "REJECTED ✗"
        impost_str = "REJECTED ✓" if impost_ok else "ACCEPTED ✗"

        print(f"  Run {run_idx + 1:>3}: "
              f"Legitimate → {legit_str:<12}  "
              f"Impostor → {impost_str}")

    far = (args.runs - impost_correct) / args.runs
    frr = (args.runs - legit_correct)  / args.runs

    print(f"\n{'═' * 60}")
    print(f"  Results: {args.runs} runs, set size={args.set_size}, T={threshold:.1f}")
    print(f"{'═' * 60}")
    print(f"  Legitimate correctly accepted : {legit_correct}/{args.runs}  "
          f"(FRR = {frr:.1%})")
    print(f"  Impostor correctly rejected   : {impost_correct}/{args.runs}  "
          f"(FAR = {far:.1%})")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()