"""
ocsvm.py

One-class classifier fusion for mouse dynamics user authentication.
Trains four one-class classifiers on legitimate user data only and
fuses their scores, following Kumar et al. (2017) "Continuous
Authentication Using One-class Classifiers and their Fusion".

The four classifiers operate on different principles:
  - OneClassSVM: finds a hyperplane boundary around the training data
  - IsolationForest: measures how easy a point is to isolate (anomalies isolate faster)
  - EllipticEnvelope: fits a Gaussian ellipse to the training data
  - LocalOutlierFactor: measures density relative to neighbors

Their errors are largely uncorrelated, so fusing their scores reduces
overall error rate compared to any single classifier.

Usage:
    python ocsvm.py --features features.csv
    python ocsvm.py --features features.csv --classifier ocsvm
    python ocsvm.py --features features.csv --personal shrey_features.csv --target_user user7
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

FEATURE_COLS = [
    'type_of_action', 'traveled_distance_pixel', 'elapsed_time',
    'direction_of_movement', 'straightness', 'num_points', 'sum_of_angles',
    'mean_curv', 'sd_curv', 'max_curv', 'min_curv',
    'mean_omega', 'sd_omega', 'max_omega', 'min_omega',
    'largest_deviation', 'dist_end_to_end_line', 'num_critical_points',
    'mean_vx', 'sd_vx', 'max_vx', 'min_vx',
    'mean_vy', 'sd_vy', 'max_vy', 'min_vy',
    'mean_v', 'sd_v', 'max_v', 'min_v',
    'mean_a', 'sd_a', 'max_a', 'min_a',
    'mean_jerk', 'sd_jerk', 'max_jerk', 'min_jerk',
    'a_beg_time'
]


def build_classifiers(nu=0.02, gamma=0.01, contamination=0.1, random_state=0):
    return {
        'ocsvm':    OneClassSVM(kernel='rbf', nu=nu, gamma=gamma),
        'iforest':  IsolationForest(contamination=contamination,
                                    random_state=random_state, n_estimators=100),
        'elliptic': EllipticEnvelope(contamination=contamination,
                                     random_state=random_state),
        'lof':      LocalOutlierFactor(novelty=True, contamination=contamination,
                                       n_neighbors=20),
    }


def normalize_scores(scores):
    """Normalize to [0,1] so all classifiers contribute equally to fusion."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min == 0:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def compute_metrics(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = float(interp1d(fpr, thresholds)(eer))
    except (ValueError, ZeroDivisionError):
        eer_threshold = float(np.median(thresholds))

    y_preds = (y_scores >= eer_threshold).astype(int)
    acc = (y_preds == y_true).mean()
    tp = ((y_preds == 1) & (y_true == 1)).sum()
    fp = ((y_preds == 1) & (y_true == 0)).sum()
    fn = ((y_preds == 0) & (y_true == 1)).sum()
    tn = ((y_preds == 0) & (y_true == 0)).sum()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {'auc': auc_score, 'eer_threshold': eer_threshold,
            'accuracy': acc, 'far': far, 'frr': frr}


def train_and_evaluate_user(target_user, df, test_size=0.33,
                             nu=0.02, gamma=0.01, contamination=0.1,
                             classifier='fusion', random_state=0):
    rng = np.random.RandomState(random_state)

    legit = df[df['userid'] == target_user][FEATURE_COLS].values
    n_legit = len(legit)
    if n_legit < 10:
        print(f"  [skip] {target_user}: not enough samples ({n_legit})")
        return None

    idx = rng.permutation(n_legit)
    n_test = max(1, int(n_legit * test_size))
    x_train      = legit[idx[n_test:]]
    x_legit_test = legit[idx[:n_test]]

    impostor_pool = df[df['userid'] != target_user][FEATURE_COLS].values
    n_impostor    = min(len(x_legit_test), len(impostor_pool))
    x_impostor_test = impostor_pool[
        rng.choice(len(impostor_pool), size=n_impostor, replace=False)
    ]

    x_test = np.vstack([x_legit_test, x_impostor_test])
    y_test = np.array([1] * len(x_legit_test) + [0] * n_impostor)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled  = scaler.transform(x_test)

    all_classifiers = build_classifiers(nu=nu, gamma=gamma,
                                        contamination=contamination,
                                        random_state=random_state)
    if classifier != 'fusion':
        all_classifiers = {classifier: all_classifiers[classifier]}

    trained = {}
    for name, model in all_classifiers.items():
        try:
            model.fit(x_train_scaled)
            trained[name] = model
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")

    if not trained:
        return None

    all_scores = []
    individual_metrics = {}
    for name, model in trained.items():
        try:
            raw  = model.decision_function(x_test_scaled)
            norm = normalize_scores(raw)
            all_scores.append(norm)
            individual_metrics[name] = compute_metrics(y_test, norm)
        except Exception as e:
            print(f"  Warning: {name} scoring failed: {e}")

    if not all_scores:
        return None

    fused_scores = np.mean(all_scores, axis=0)
    metrics = compute_metrics(y_test, fused_scores)
    metrics['user']           = target_user
    metrics['n_train']        = len(x_train)
    metrics['n_test_legit']   = len(x_legit_test)
    metrics['n_test_impostor'] = n_impostor
    metrics['individual']     = individual_metrics

    return trained, scaler, metrics


def evaluate_personal_session(trained_models, scaler, path, personal_user, threshold):
    df = pd.read_csv(path)
    x  = df[[c for c in FEATURE_COLS if c in df.columns]].values
    x_scaled = scaler.transform(x)

    all_scores = []
    for name, model in trained_models.items():
        norm = normalize_scores(model.decision_function(x_scaled))
        all_scores.append(norm)

    fused      = np.mean(all_scores, axis=0)
    mean_score = fused.mean()
    accepted   = mean_score >= threshold

    print(f"\n── Personal Session Evaluation ──────────────────────────────")
    print(f"  File:          {path}")
    print(f"  User:          {personal_user}")
    print(f"  Actions:       {len(x)}")
    print(f"  Mean score:    {mean_score:.4f}")
    print(f"  EER threshold: {threshold:.4f}")
    print(f"  Decision:      {'✓ ACCEPTED (legitimate)' if accepted else '✗ REJECTED (impostor)'}")
    print(f"\n  Per-classifier scores:")
    for name, model in trained_models.items():
        norm = normalize_scores(model.decision_function(x_scaled))
        print(f"    {name:<12} {norm.mean():.4f}")

    return mean_score, accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features',    required=True)
    parser.add_argument('--classifier',  default='fusion',
                        choices=['fusion', 'ocsvm', 'iforest', 'elliptic', 'lof'])
    parser.add_argument('--nu',          type=float, default=0.02)
    parser.add_argument('--gamma',       type=float, default=0.01)
    parser.add_argument('--test_size',   type=float, default=0.33)
    parser.add_argument('--personal',    default=None)
    parser.add_argument('--personal_user', default='shrey')
    parser.add_argument('--target_user', default=None)
    parser.add_argument('--save_models', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)

    users = [args.target_user] if args.target_user else sorted(df['userid'].unique())

    results = []
    trained_models_all = {}

    for user in users:
        print(f"\n{'='*50}")
        print(f"Training [{args.classifier}] for user: {user}")
        print(f"{'='*50}")

        result = train_and_evaluate_user(
            target_user=user, df=df,
            test_size=args.test_size,
            nu=args.nu, gamma=args.gamma,
            contamination=args.nu,
            classifier=args.classifier,
        )
        if result is None:
            continue

        trained, scaler, metrics = result
        trained_models_all[user] = (trained, scaler, metrics)

        print(f"  Train:    {metrics['n_train']}  |  "
              f"Test legit: {metrics['n_test_legit']}  |  "
              f"Test impostor: {metrics['n_test_impostor']}")

        if args.classifier == 'fusion' and 'individual' in metrics:
            print(f"  Individual AUCs:")
            for name, m in metrics['individual'].items():
                print(f"    {name:<12} AUC: {m['auc']:.4f}  "
                      f"FAR: {m['far']:.4f}  FRR: {m['frr']:.4f}")

        print(f"  {'Fused' if args.classifier == 'fusion' else args.upper()} "
              f"AUC: {metrics['auc']:.4f}  "
              f"Acc: {metrics['accuracy']:.4f}  "
              f"FAR: {metrics['far']:.4f}  FRR: {metrics['frr']:.4f}")

        results.append(metrics)

        if args.save_models:
            os.makedirs('checkpoints_ocsvm', exist_ok=True)
            joblib.dump(trained, f'checkpoints_ocsvm/models_{user}.pkl')
            joblib.dump(scaler,  f'checkpoints_ocsvm/scaler_{user}.pkl')

    if results:
        print(f"\n[{args.classifier}]")
        print(f"\n{'User':<12} {'AUC':>8} {'EER Thr':>10} {'Accuracy':>10} "
              f"{'FAR':>8} {'FRR':>8}")
        print('-' * 62)
        aucs, accs, fars, frrs = [], [], [], []
        for r in results:
            aucs.append(r['auc']); accs.append(r['accuracy'])
            fars.append(r['far']); frrs.append(r['frr'])
            print(f"{str(r['user']):<12} {r['auc']:>8.4f} "
                  f"{r['eer_threshold']:>10.4f} {r['accuracy']:>10.4f} "
                  f"{r['far']:>8.4f} {r['frr']:>8.4f}")
        print('-' * 62)
        print(f"{'Mean':<12} {np.mean(aucs):>8.4f} {'':>10} "
              f"{np.mean(accs):>10.4f} {np.mean(fars):>8.4f} "
              f"{np.mean(frrs):>8.4f}")

    if args.personal and args.target_user and args.target_user in trained_models_all:
        trained, scaler, metrics = trained_models_all[args.target_user]
        evaluate_personal_session(trained, scaler, args.personal,
                                  args.personal_user, metrics['eer_threshold'])
    elif args.personal:
        print("\nNote: --personal requires --target_user.")


if __name__ == '__main__':
    main()