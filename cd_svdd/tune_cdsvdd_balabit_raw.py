import sys
import os
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

DATA_DIR   = "balabit_dataset/training_files"
OUTPUT_FILE = "tuning_results.txt"

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULTS = {
    "nu":           0.1,
    "output_dim":   8,
    "lr":           1e-3,
    "window_size":  200,
    "stride_frac":  0.5,
    "n_layers":     4,
    "kernel_size":  3,
    "channels":     32,
    "qp_subsample": 256,
    "n_epochs":     50,
}



# ── Parameter grids ───────────────────────────────────────────────────────────

GRIDS = {
    "nu":           [0.01, 0.05, 0.1, 0.2, 0.3],
    "output_dim":   [4, 8, 16, 32],
    "lr":           [1e-4, 1e-3, 1e-2],
    "window_size":  [100, 200, 300, 400],
    "stride_frac":  [0.25, 0.5, 0.75, 1.0],
    "n_layers":     [2, 4, 6, 8],
    "kernel_size":  [3, 5, 7],
    "channels":     [16, 32, 64, 128],
    "qp_subsample": [64, 128, 256, 512],
    "n_epochs":     [25, 50, 100, 200],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_session_files(user_dir):
    return sorted([
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if os.path.isfile(os.path.join(user_dir, f))
    ])


def run_evaluation(params):
    window_size  = params["window_size"]
    stride       = max(1, int(params["stride_frac"] * window_size))
    nu           = params["nu"]
    output_dim   = params["output_dim"]
    lr           = params["lr"]
    n_layers     = params["n_layers"]
    kernel_size  = params["kernel_size"]
    channels     = params["channels"]
    qp_subsample = params["qp_subsample"]
    n_epochs     = params["n_epochs"]

    all_results = []

    for user in BALABIT_USERS:
        user_dir  = os.path.join(DATA_DIR, user)
        all_files = get_session_files(user_dir)
        all_windows = extract_raw_windows(all_files, window_size, stride)

        if len(all_windows) < 8:
            print(f"  {user}: not enough windows, skipping", flush=True)
            continue

        n_total = len(all_windows)
        n_test  = max(1, int(n_total * 0.25))
        n_train = n_total - n_test

        train_windows = np.array(all_windows[:n_train], dtype=np.float32)
        test_windows  = np.array(all_windows[n_train:],  dtype=np.float32)

        effective_qp = min(qp_subsample, n_train)

        net = TCNNetwork(
            input_dim=INPUT_DIM,
            channels=channels,
            n_layers=n_layers,
            kernel_size=kernel_size,
            output_dim=output_dim,
        )

        model = CDSVDD(
            input_dim=INPUT_DIM,
            output_dim=output_dim,
            nu=nu,
            n_epochs=n_epochs,
            lr=lr,
            qp_subsample=effective_qp,
        )
        model.net = net

        print(f"  Training {user}...", flush=True)
        model.fit(train_windows)

        legit_scores   = model.score(test_windows)
        legit_accepted = int(np.sum(legit_scores <= 0))
        legit_scored   = len(legit_scores)

        impostor_accepted   = 0
        impostor_scored     = 0
        all_impostor_scores = []

        for other_user in BALABIT_USERS:
            if other_user == user:
                continue
            other_files   = get_session_files(os.path.join(DATA_DIR, other_user))
            other_windows = extract_raw_windows(other_files, window_size, stride)
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

        all_results.append({
            "user": user,
            "legit_accepted": legit_accepted, "legit_scored": legit_scored,
            "impostor_rejected": impostor_rejected, "impostor_scored": impostor_scored,
            "far": far, "frr": frr, "acc": acc, "auc": auc,
            "all_scores": -all_scores,
            "all_labels": all_labels,
        })

    return all_results


def format_results(param_name, param_value, results):
    lines = []
    lines.append(f"\n--- {param_name} = {param_value} ---")

    if not results:
        lines.append("  No results.")
        return "\n".join(lines)

    lines.append(f"  {'User':<12} {'FAR':>8} {'FRR':>8} {'Accuracy':>10} {'AUC':>8}")
    lines.append(f"  {'-'*50}")

    for r in results:
        lines.append(
            f"  {r['user']:<12} {r['far']*100:>7.1f}% "
            f"{r['frr']*100:>7.1f}% {r['acc']*100:>9.1f}% {r['auc']:>8.4f}"
        )

    lines.append(f"  {'-'*50}")

    mean_far = sum(r["far"] for r in results) / len(results)
    mean_frr = sum(r["frr"] for r in results) / len(results)
    mean_acc = sum(r["acc"] for r in results) / len(results)
    mean_auc = sum(r["auc"] for r in results if not np.isnan(r["auc"])) \
               / sum(1 for r in results if not np.isnan(r["auc"]))

    tla = sum(r["legit_accepted"]    for r in results)
    tls = sum(r["legit_scored"]      for r in results)
    tir = sum(r["impostor_rejected"] for r in results)
    tis = sum(r["impostor_scored"]   for r in results)

    micro_all_scores = np.concatenate([r["all_scores"] for r in results])
    micro_all_labels = np.concatenate([r["all_labels"] for r in results])
    micro_auc = roc_auc_score(micro_all_labels, micro_all_scores) \
                if len(np.unique(micro_all_labels)) == 2 else float("nan")

    micro_frr = 1 - tla / tls          if tls > 0 else 0.0
    micro_far = (tis - tir) / tis      if tis > 0 else 0.0
    micro_acc = (tla + tir) / (tls + tis) if (tls + tis) > 0 else 0.0

    lines.append(
        f"  {'Mean':<12} {mean_far*100:>7.1f}% "
        f"{mean_frr*100:>7.1f}% {mean_acc*100:>9.1f}% {mean_auc:>8.4f}"
    )
    lines.append(
        f"  {'Micro':<12} {micro_far*100:>7.1f}% "
        f"{micro_frr*100:>7.1f}% {micro_acc*100:>9.1f}% {micro_auc:>8.4f}"
    )
    lines.append(f"\n  Total legitimate: {tla}/{tls} accepted")
    lines.append(f"  Total impostors:  {tir}/{tis} rejected")

    return "\n".join(lines)


def write(f, text):
    print(text, flush=True)
    f.write(text + "\n")
    f.flush()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_param", default=None, help="Parameter name to resume from")
    parser.add_argument("--start_value", default=None, help="Parameter value to resume from (as string)")
    args = parser.parse_args()

    mode = "a" if (args.start_param or args.start_value) else "w"

    with open(OUTPUT_FILE, mode) as f:
        if mode == "w":
            write(f, f"CD-SVDD TCN Tuning Results")
            write(f, f"Defaults: {DEFAULTS}\n")

        param_names = list(GRIDS.keys())
        start_param = args.start_param
        start_value = args.start_value

        skipping_param = start_param is not None

        for param_name, grid in GRIDS.items():
            if skipping_param:
                if param_name != start_param:
                    continue
                skipping_param = False

            header = f"\n{'='*62}\nParameter: {param_name}\n{'='*62}"
            write(f, header)

            skipping_value = start_value is not None and param_name == start_param

            for value in grid:
                if skipping_value:
                    if str(value) != start_value:
                        continue
                    skipping_value = False

                write(f, f"\nRunning {param_name} = {value}...")

                params = dict(DEFAULTS)
                params[param_name] = value

                results = run_evaluation(params)
                block = format_results(param_name, value, results)
                write(f, block)

        write(f, "\nTuning complete.")


if __name__ == "__main__":
    main()