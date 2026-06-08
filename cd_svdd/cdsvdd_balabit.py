import sys
import os
print("Starting imports...", flush=True)
import argparse
import numpy as np
print("numpy ok", flush=True)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
print("sklearn ok", flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Importing extract_features_scroll...", flush=True)
from measurements.extract_features_scroll import (
    extract_session_features, MORE_SCROLL_COLS, DIR_SCROLL_COLS
)
print("extract_features_scroll ok", flush=True)

print("Importing CDSVDD...", flush=True)
from cd_svdd_model import CDSVDD
print("CDSVDD ok", flush=True)

BALABIT_USERS = [
    "user7", "user9", "user12", "user15", "user16",
    "user20", "user21", "user23", "user29", "user35",
]

DATA_DIR = "balabit_dataset/training_files"

HOLISTIC_COLS = [
    "type_of_action", "traveled_distance_pixel", "elapsed_time",
    "direction_of_movement", "straightness", "num_points", "sum_of_angles",
    "mean_curv", "sd_curv", "max_curv", "min_curv",
    "mean_omega", "sd_omega", "max_omega", "min_omega",
    "largest_deviation", "dist_end_to_end_line", "num_critical_points",
    "mean_vx", "sd_vx", "max_vx", "min_vx",
    "mean_vy", "sd_vy", "max_vy", "min_vy",
    "mean_v", "sd_v", "max_v", "min_v",
    "mean_a", "sd_a", "max_a", "min_a",
    "mean_jerk", "sd_jerk", "max_jerk", "min_jerk", "a_beg_time",
]

SCROLL_COLS = [
    "scroll_count", "scroll_rate", "scroll_ratio", "scroll_up_ratio",
    "scroll_dur_mean", "scroll_dur_std",
    "scroll_burst_count", "scroll_burst_dur_mean", "scroll_burst_len_mean",
]

FEATURE_COLS = HOLISTIC_COLS + SCROLL_COLS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu",            type=float, default=0.1)
    parser.add_argument("--output_dim",    type=int,   default=8)
    parser.add_argument("--hidden_dim",    type=int,   default=32)
    parser.add_argument("--n_epochs",      type=int,   default=50)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--window_size",   type=int,   default=50)
    parser.add_argument("--held_out_frac", type=float, default=0.25)
    parser.add_argument("--more_scroll",   action="store_true")
    parser.add_argument("--dir_scroll",    action="store_true")
    return parser.parse_args()


def get_session_files(user_dir):
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def extract_all_windows(session_files, user_id, window_size, feature_cols,
                        more_scroll=False, dir_scroll=False):
    all_vecs = []
    for path in session_files:
        try:
            df = extract_session_features(path, user_id, window_size=window_size,
                                          more_scroll=more_scroll, dir_scroll=dir_scroll)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
            if len(df) == 0:
                continue
            for _, grp in df.groupby("session"):
                rows = grp[feature_cols].values
                if len(rows) >= 1:
                    all_vecs.append(rows.mean(axis=0))
        except Exception as e:
            print(f"  [!] {os.path.basename(path)}: {e}", flush=True)
    return all_vecs


def main():
    args = parse_args()

    feature_cols = (HOLISTIC_COLS + SCROLL_COLS
                    + (MORE_SCROLL_COLS if args.more_scroll else [])
                    + (DIR_SCROLL_COLS  if args.dir_scroll  else []))

    print(f"nu={args.nu}  output_dim={args.output_dim}  "
          f"n_epochs={args.n_epochs}  window_size={args.window_size}")
    print(f"Feature vector: {len(HOLISTIC_COLS)} holistic"
          f" + {len(SCROLL_COLS)} scroll"
          + (f" + {len(MORE_SCROLL_COLS)} deltaY scroll" if args.more_scroll else "")
          + (f" + {len(DIR_SCROLL_COLS)} dir scroll"     if args.dir_scroll  else "")
          + f" = {len(feature_cols)} total\n", flush=True)

    all_results = []

    for user in BALABIT_USERS:
        print(f"{'='*50}")
        print(f"User: {user}")
        print(f"{'='*50}", flush=True)

        user_dir  = os.path.join(DATA_DIR, user)
        all_files = get_session_files(user_dir)

        print(f"  Extracting windows from {len(all_files)} session files...", flush=True)
        all_vecs  = extract_all_windows(all_files, user, args.window_size,
                                        feature_cols, args.more_scroll, args.dir_scroll)

        if len(all_vecs) < 8:
            print(f"  Not enough windows ({len(all_vecs)}), skipping", flush=True)
            continue

        n_total = len(all_vecs)
        n_test  = max(1, int(n_total * args.held_out_frac))
        n_train = n_total - n_test

        train_samples = np.array(all_vecs[:n_train], dtype=np.float32)
        test_samples  = np.array(all_vecs[n_train:],  dtype=np.float32)

        print(f"  Total windows: {n_total}  |  Train: {n_train}  |  Test: {n_test}", flush=True)

        scaler = StandardScaler()
        train_norm = scaler.fit_transform(train_samples)
        test_norm  = scaler.transform(test_samples)

        print(f"  Training CD-SVDD...", flush=True)
        model = CDSVDD(
            input_dim=len(feature_cols),
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            nu=args.nu,
            n_epochs=args.n_epochs,
            lr=args.lr,
        )
        model.fit(train_norm)

        print(f"  Evaluating on legitimate held-out windows...", flush=True)
        legit_scores   = model.score(test_norm)
        legit_accepted = int(np.sum(legit_scores <= 0))
        legit_scored   = len(legit_scores)

        print(f"\n  Legitimate held-out ({legit_scored} windows):")
        print(f"    Scores: min={legit_scores.min():+.4f}  "
              f"mean={legit_scores.mean():+.4f}  max={legit_scores.max():+.4f}")
        print(f"    Accepted: {legit_accepted}/{legit_scored}", flush=True)

        impostor_accepted   = 0
        impostor_scored     = 0
        all_impostor_scores = []

        for other_user in BALABIT_USERS:
            if other_user == user:
                continue
            print(f"  Evaluating impostor: {other_user}...", flush=True)
            other_files = get_session_files(os.path.join(DATA_DIR, other_user))
            other_vecs  = extract_all_windows(other_files, other_user, args.window_size,
                                               feature_cols, args.more_scroll, args.dir_scroll)
            if len(other_vecs) == 0:
                continue

            other_norm  = scaler.transform(np.array(other_vecs, dtype=np.float32))
            imp_scores  = model.score(other_norm)

            impostor_scored   += len(imp_scores)
            impostor_accepted += int(np.sum(imp_scores <= 0))
            all_impostor_scores.extend(imp_scores.tolist())

        impostor_rejected = impostor_scored - impostor_accepted

        all_scores = np.concatenate([legit_scores, np.array(all_impostor_scores)])
        all_labels = np.concatenate([
            np.ones(len(legit_scores)),
            np.zeros(len(all_impostor_scores)),
        ])
        auc = roc_auc_score(all_labels, -all_scores) \
              if len(np.unique(all_labels)) == 2 else float("nan")

        frr = 1 - legit_accepted / legit_scored    if legit_scored    > 0 else 0.0
        far = impostor_accepted  / impostor_scored  if impostor_scored > 0 else 0.0
        acc = (legit_accepted + impostor_rejected) / (legit_scored + impostor_scored) \
              if (legit_scored + impostor_scored) > 0 else 0.0

        print(f"\n  Impostors:  {impostor_rejected}/{impostor_scored} rejected")
        print(f"  FAR: {far*100:.1f}%  FRR: {frr*100:.1f}%  "
              f"Accuracy: {acc*100:.1f}%  AUC: {auc:.4f}\n", flush=True)

        all_results.append({
            "user": user,
            "legit_accepted": legit_accepted, "legit_scored": legit_scored,
            "impostor_rejected": impostor_rejected, "impostor_scored": impostor_scored,
            "far": far, "frr": frr, "acc": acc, "auc": auc,
            "all_scores": -all_scores,
            "all_labels": all_labels,
        })

    if not all_results:
        print("No results.")
        return

    mean_far = sum(r["far"] for r in all_results) / len(all_results)
    mean_frr = sum(r["frr"] for r in all_results) / len(all_results)
    mean_acc = sum(r["acc"] for r in all_results) / len(all_results)
    mean_auc = sum(r["auc"] for r in all_results if not np.isnan(r["auc"])) \
               / sum(1 for r in all_results if not np.isnan(r["auc"]))

    tla = sum(r["legit_accepted"]    for r in all_results)
    tls = sum(r["legit_scored"]      for r in all_results)
    tir = sum(r["impostor_rejected"] for r in all_results)
    tis = sum(r["impostor_scored"]   for r in all_results)

    micro_all_scores = np.concatenate([r["all_scores"] for r in all_results])
    micro_all_labels = np.concatenate([r["all_labels"] for r in all_results])
    micro_auc = roc_auc_score(micro_all_labels, micro_all_scores) \
                if len(np.unique(micro_all_labels)) == 2 else float("nan")

    micro_frr = 1 - tla / tls          if tls > 0 else 0.0
    micro_far = (tis - tir) / tis      if tis > 0 else 0.0
    micro_acc = (tla + tir) / (tls + tis) if (tls + tis) > 0 else 0.0

    print(f"\n{'='*62}")
    print(f"  Aggregate Results ({len(all_results)} users)")
    print(f"{'='*62}")
    print(f"  {'User':<12} {'FAR':>8} {'FRR':>8} {'Accuracy':>10} {'AUC':>8}")
    print(f"  {'-'*50}")
    for r in all_results:
        print(f"  {r['user']:<12} {r['far']*100:>7.1f}% "
              f"{r['frr']*100:>7.1f}% {r['acc']*100:>9.1f}% {r['auc']:>8.4f}")
    print(f"  {'-'*50}")
    print(f"  {'Mean':<12} {mean_far*100:>7.1f}% "
          f"{mean_frr*100:>7.1f}% {mean_acc*100:>9.1f}% {mean_auc:>8.4f}")
    print(f"  {'Micro':<12} {micro_far*100:>7.1f}% "
          f"{micro_frr*100:>7.1f}% {micro_acc*100:>9.1f}% {micro_auc:>8.4f}")
    print(f"\n  Total legitimate: {tla}/{tls} accepted")
    print(f"  Total impostors:  {tir}/{tis} rejected")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()