"""
ocsvm_fusion.py

Evaluates and fuses two independent pipelines:
  1. 39-feature pipeline  — per-action OCSVM (same as ocsvm.py)
  2. Zheng pipeline       — per-session leave-one-out OCSVM

Fusion: for each user, both pipelines produce a single AUC score.
To fuse, we combine their normalized decision scores at the action/session
level before computing final metrics.

Usage:
    python ocsvm_fusion.py --features balabit_features.csv --zheng zheng_features.csv
    python ocsvm_fusion.py --features balabit_features.csv --zheng zheng_features.csv --mode zheng_only
    python ocsvm_fusion.py --features balabit_features.csv --zheng zheng_features.csv --mode features_only
"""
import argparse, numpy as np, pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

FEATURE_COLS_39 = [
    "type_of_action","traveled_distance_pixel","elapsed_time",
    "direction_of_movement","straightness","num_points","sum_of_angles",
    "mean_curv","sd_curv","max_curv","min_curv",
    "mean_omega","sd_omega","max_omega","min_omega",
    "largest_deviation","dist_end_to_end_line","num_critical_points",
    "mean_vx","sd_vx","max_vx","min_vx",
    "mean_vy","sd_vy","max_vy","min_vy",
    "mean_v","sd_v","max_v","min_v",
    "mean_a","sd_a","max_a","min_a",
    "mean_jerk","sd_jerk","max_jerk","min_jerk","a_beg_time"
]
FEATURE_COLS_ZHENG = [f"ca_{i}" for i in range(180)] + [f"cd_{i}" for i in range(35)]

# ── Helpers ───────────────────────────────────────────────────────────────

def normalize_scores(s):
    lo,hi = s.min(),s.max()
    return (s-lo)/(hi-lo) if hi-lo>1e-9 else np.zeros_like(s)

def compute_metrics(y_true, y_scores):
    fpr,tpr,thr = roc_curve(y_true, y_scores)
    a = auc(fpr,tpr)
    try:
        eer = brentq(lambda x:1.-x-interp1d(fpr,tpr)(x),0.,1.)
        et  = float(interp1d(fpr,thr)(eer))
    except:
        et = float(np.median(thr))
    yp = (y_scores>=et).astype(int)
    tp=((yp==1)&(y_true==1)).sum(); fp=((yp==1)&(y_true==0)).sum()
    fn=((yp==0)&(y_true==1)).sum(); tn=((yp==0)&(y_true==0)).sum()
    far=fp/(fp+tn) if fp+tn>0 else 0.
    frr=fn/(fn+tp) if fn+tp>0 else 0.
    return {"auc":a,"eer_thr":et,"acc":(yp==y_true).mean(),"far":far,"frr":frr}

def fit_fused(x_train, x_test, nu, gamma):
    """Fit 4-classifier fusion, return normalized fused scores."""
    sc = StandardScaler()
    xtr = sc.fit_transform(x_train)
    xte = sc.transform(x_test)
    ms = {
        "ocsvm":    OneClassSVM(kernel="rbf",nu=nu,gamma=gamma),
        "iforest":  IsolationForest(contamination=nu,n_estimators=100,random_state=0),
        "elliptic": EllipticEnvelope(contamination=nu,random_state=0),
        "lof":      LocalOutlierFactor(novelty=True,contamination=nu,n_neighbors=20),
    }
    scores = []
    for nm,m in ms.items():
        try:
            m.fit(xtr)
            scores.append(normalize_scores(m.decision_function(xte)))
        except Exception as e:
            print(f"    {nm}: {e}")
    return np.mean(scores,axis=0) if scores else None

# ── 39-feature pipeline: per-action evaluation (correct) ──────────────────

def pipeline_39(user, df, rng, test_size, nu, gamma):
    """
    Identical to ocsvm.py — per-action train/test split.
    Returns (y_true, y_scores) over all test actions.
    """
    legit = df[df["userid"]==user][FEATURE_COLS_39].values
    n = len(legit)
    if n < 10: return None, None
    idx = rng.permutation(n)
    nt  = max(1,int(n*test_size))
    x_train      = legit[idx[nt:]]
    x_legit_test = legit[idx[:nt]]
    imp = df[df["userid"]!=user][FEATURE_COLS_39].values
    ni  = min(len(x_legit_test),len(imp))
    if ni==0: return None,None
    x_imp_test = imp[rng.choice(len(imp),ni,replace=False)]
    x_test = np.vstack([x_legit_test, x_imp_test])
    y_test = np.array([1]*len(x_legit_test) + [0]*ni)
    scores = fit_fused(x_train, x_test, nu, gamma)
    if scores is None: return None, None
    return y_test, scores

# ── Zheng pipeline: leave-one-out at session level ────────────────────────

def pipeline_zheng(user, df, rng, nu, gamma):
    """
    Leave-one-out cross validation over sessions.
    For each session of the target user:
      - Train on all OTHER sessions of that user
      - Test: left-out session (legitimate) vs one random session
              from each other user (impostors)
    Accumulates one score per legitimate session and one per impostor session.
    Returns (y_true, y_scores).
    """
    legit = df[df["userid"]==user][FEATURE_COLS_ZHENG].values
    n = len(legit)
    if n < 3: return None, None
    imp_pool = df[df["userid"]!=user][FEATURE_COLS_ZHENG].values
    if len(imp_pool) == 0: return None, None
    all_scores = []
    all_labels = []
    for i in range(n):
        # leave session i out as the test legitimate sample
        x_train = np.delete(legit, i, axis=0)
        if len(x_train) < 2: continue
        x_legit_test = legit[i:i+1]
        # sample one impostor session per fold
        imp_idx = rng.choice(len(imp_pool), size=min(n, len(imp_pool)), replace=False)
        x_imp_test = imp_pool[imp_idx]
        x_test = np.vstack([x_legit_test, x_imp_test])
        y_test = np.array([1] + [0]*len(x_imp_test))
        scores = fit_fused(x_train, x_test, nu, gamma)
        if scores is None: continue
        all_scores.extend(scores.tolist())
        all_labels.extend(y_test.tolist())
    if not all_scores: return None, None
    return np.array(all_labels), np.array(all_scores)

# ── Per-user evaluation ───────────────────────────────────────────────────

def evaluate_user(user, df39, dfzh, test_size, nu, gamma, mode):
    rng = np.random.RandomState(0)
    res = {}
    y39, s39   = pipeline_39(user, df39, rng, test_size, nu, gamma) if mode in ("features_only","fusion") else (None,None)
    yzh, szh   = pipeline_zheng(user, dfzh, rng, nu, gamma)         if mode in ("zheng_only","fusion")    else (None,None)
    if y39 is not None:
        res["features_39"] = compute_metrics(y39, s39)
    if yzh is not None:
        res["zheng"] = compute_metrics(yzh, szh)
    if mode == "fusion" and y39 is not None and yzh is not None:
        # to fuse, we need comparable score sets
        # use the zheng LOO scores (one per session) as session-level signal
        # and the 39-feature scores (per action) as action-level signal
        # report them side by side — direct score fusion is not meaningful
        # across different granularities, so fusion AUC = mean of both AUCs
        auc_39  = res["features_39"]["auc"]
        auc_zh  = res["zheng"]["auc"]
        far_39  = res["features_39"]["far"]
        far_zh  = res["zheng"]["far"]
        frr_39  = res["features_39"]["frr"]
        frr_zh  = res["zheng"]["frr"]
        res["fusion"] = {
            "auc":     (auc_39 + auc_zh) / 2,
            "eer_thr": 0.0,
            "acc":     (res["features_39"]["acc"] + res["zheng"]["acc"]) / 2,
            "far":     (far_39 + far_zh) / 2,
            "frr":     (frr_39 + frr_zh) / 2,
        }
    return res

# ── Printing ──────────────────────────────────────────────────────────────

def print_table(label, rows):
    print(f"\n{"="*62}\n  {label}\n{"="*62}")
    print(f"{"User":<12} {"AUC":>8} {"EER Thr":>10} {"Accuracy":>10} {"FAR":>8} {"FRR":>8}")
    print("-"*62)
    aucs,accs,fars,frrs=[],[],[],[]
    for u,m in rows:
        aucs.append(m["auc"]); accs.append(m["acc"])
        fars.append(m["far"]); frrs.append(m["frr"])
        print(f"{str(u):<12} {m['auc']:>8.4f} {m['eer_thr']:>10.4f} {m['acc']:>10.4f} {m['far']:>8.4f} {m['frr']:>8.4f}")
    print("-"*62)
    print(f"{"Mean":<12} {np.mean(aucs):>8.4f} {"":>10} {np.mean(accs):>10.4f} {np.mean(fars):>8.4f} {np.mean(frrs):>8.4f}")

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features",  required=True, help="39-feature CSV")
    p.add_argument("--zheng",     required=True, help="Zheng 215-feature CSV")
    p.add_argument("--mode",      default="fusion",
                   choices=["fusion","features_only","zheng_only"])
    p.add_argument("--nu",        type=float, default=0.02)
    p.add_argument("--gamma",     type=float, default=0.01)
    p.add_argument("--test_size", type=float, default=0.33)
    p.add_argument("--users",     nargs="+", default=None)
    args = p.parse_args()
    df39 = pd.read_csv(args.features).replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURE_COLS_39)
    dfzh = pd.read_csv(args.zheng).replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURE_COLS_ZHENG)
    users = args.users or sorted(df39["userid"].unique())
    r39,rzh,rfused = [],[],[]
    for u in users:
        print(f"Evaluating {u}...")
        r = evaluate_user(u,df39,dfzh,args.test_size,args.nu,args.gamma,args.mode)
        if "features_39" in r: r39.append((u,r["features_39"]))
        if "zheng"       in r: rzh.append((u,r["zheng"]))
        if "fusion"      in r: rfused.append((u,r["fusion"]))
    if r39:    print_table("39-Feature Pipeline (per-action, correct)", r39)
    if rzh:    print_table("Zheng 215-dim (leave-one-out per session)", rzh)
    if rfused: print_table("Fusion: mean AUC of both pipelines", rfused)

if __name__ == "__main__":
    main()