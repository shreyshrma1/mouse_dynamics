import sys
import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_raw_windows import extract_raw_windows, INPUT_DIM
from backbone_tcn import TCNNetwork
from cd_svdd_model import CDSVDD

BALABIT_USERS = [
    "user7", "user9", "user12", "user15", "user16",
    "user20", "user21", "user23", "user29", "user35",
]

DATA_DIR = "balabit_dataset/training_files"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu",           type=float, default=0.3)
    parser.add_argument("--output_dim",   type=int,   default=8)
    parser.add_argument("--channels",     type=int,   default=32)
    parser.add_argument("--n_layers",     type=int,   default=4)
    parser.add_argument("--n_epochs",     type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--window_size",  type=int,   default=200)
    parser.add_argument("--stride",       type=int,   default=100)
    parser.add_argument("--held_out_frac",type=float, default=0.25)
    parser.add_argument("--qp_subsample", type=int,   default=256)
    return parser.parse_args()


def get_session_files(user_dir):
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def main():
    args = parse_args()

    print(f"nu={args.nu}  output_dim={args.output_dim}  channels={args.channels}  "
          f"n_layers={args.n_layers}  n_epochs={args.n_epochs}  "
          f"window_size={args.window_size}\n", flush=True)

    all_results = []

    for user in BALABIT_USERS:
        print(f"{'='*50}")
        print(f"User: {user}")
        print(f"{'='*50}", flush=True)

        user_dir  = os.path.join(DATA_DIR, user)
        all_files = get_session_files(user_dir)

        print(f"  Extracting raw windows from {len(all_files)} session files...", flush=True)
        all_windows = extract_raw_windows(all_files, args.window_size, args.stride)

        if len(all_windows) < 8:
            print(f"  Not enough windows ({len(all_windows)}), skipping", flush=True)
            continue

        n_total = len(all_windows)
        n_test  = max(1, int(n_total * args.held_out_frac))
        n_train = n_total - n_test

        train_windows = np.array(all_windows[:n_train], dtype=np.float32)
        test_windows  = np.array(all_windows[n_train:],  dtype=np.float32)

        print(f"  Total windows: {n_total}  |  Train: {n_train}  |  Test: {n_test}", flush=True)

        net = TCNNetwork(
            input_dim=INPUT_DIM,
            channels=args.channels,
            n_layers=args.n_layers,
            output_dim=args.output_dim,
        )

        effective_qp = min(args.qp_subsample, n_train)

        model = CDSVDD(
            input_dim=INPUT_DIM,
            output_dim=args.output_dim,
            nu=args.nu,
            n_epochs=args.n_epochs,
            lr=args.lr,
            qp_subsample=effective_qp,
        )
        model.net = net

        model.fit(train_windows)

        print(f"  Evaluating on legitimate held-out windows...", flush=True)
        legit_scores   = model.score(test_windows)
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
            other_files   = get_session_files(os.path.join(DATA_DIR, other_user))
            other_windows = extract_raw_windows(other_files, args.window_size, args.stride)
            if len(other_windows) == 0:
                continue

            other_arr  = np.array(other_windows, dtype=np.float32)
            imp_scores = model.score(other_arr)

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