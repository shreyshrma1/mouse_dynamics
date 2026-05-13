"""
ocsvm_fusion_v2.py

True score-level fusion of 39-feature and Zheng pipelines.
Both pipelines now operate at SESSION granularity:

  39-feature: per-action scores averaged within each session
  Zheng:      per-session histogram scores (leave-one-out)

Both produce one score per session, so they can be directly
fused by averaging before calling roc_curve.

Requires extract_features.py to have been run with the session
column included (updated version adds session = filename).

Usage:
    python ocsvm_fusion_v2.py --features balabit_features.csv --zheng zheng_features.csv
    python ocsvm_fusion_v2.py --features balabit_features.csv --zheng zheng_features.csv --mode zheng_only
    python ocsvm_fusion_v2.py --features balabit_features.csv --zheng zheng_features.csv --mode features_only
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

# ── 39-feature pipeline: LOO at session level ────────────────────────────

def pipeline_39_session(user, df, rng, nu, gamma):
    """
    Leave-one-out over sessions for the 39-feature pipeline.
    For each session of the target user:
      - Train OCSVM on all actions from all OTHER sessions
      - Score each action in the left-out session, average -> session score
      - Score actions from impostor sessions, average -> impostor session scores
    Returns (y_true, session_scores) — one score per session.
    """
    if "session" not in df.columns:
        print("  Error: features CSV missing session column. Re-run extract_features.py.")
        return None, None
    user_df = df[df["userid"]==user]
    sessions = sorted(user_df["session"].unique())
    n = len(sessions)
    if n < 3: return None, None
    imp_df = df[df["userid"]!=user]
    imp_sessions = sorted(imp_df["session"].unique())
    if len(imp_sessions) == 0: return None, None
    all_scores = []
    all_labels = []
    for i, left_out in enumerate(sessions):
        # train on all actions from all other sessions of this user
        train_mask = (df["userid"]==user) & (df["session"]!=left_out)
        x_train = df[train_mask][FEATURE_COLS_39].values
        if len(x_train) < 10: continue
        # test: left-out session actions (legitimate)
        x_legit = user_df[user_df["session"]==left_out][FEATURE_COLS_39].values
        # test: sample n impostor sessions from other users
        sampled_imp_sessions = rng.choice(imp_sessions, size=min(n, len(imp_sessions)), replace=False)
        x_test_parts = [x_legit]
        session_boundaries = [len(x_legit)]
        for imp_sess in sampled_imp_sessions:
            x_imp = imp_df[imp_df["session"]==imp_sess][FEATURE_COLS_39].values
            if len(x_imp) > 0:
                x_test_parts.append(x_imp)
                session_boundaries.append(session_boundaries[-1] + len(x_imp))
        x_test = np.vstack(x_test_parts)
        raw = fit_fused(x_train, x_test, nu, gamma)
        if raw is None: continue
        # average action scores within each session -> one score per session
        legit_score = raw[:len(x_legit)].mean()
        all_scores.append(legit_score)
        all_labels.append(1)
        offset = len(x_legit)
        for imp_sess, x_imp_part in zip(sampled_imp_sessions, x_test_parts[1:]):
            n_imp = len(x_imp_part)
            imp_score = raw[offset:offset+n_imp].mean()
            all_scores.append(imp_score)
            all_labels.append(0)
            offset += n_imp
    if not all_scores: return None, None
    return np.array(all_labels), np.array(all_scores)

# ── Zheng pipeline: LOO at session level ─────────────────────────────────

def pipeline_zheng_session(user, df, rng, nu, gamma):
    """
    Leave-one-out over sessions for the Zheng pipeline.
    Each row in df is already one session histogram.
    Returns (y_true, session_scores).
    """
    legit = df[df["userid"]==user][FEATURE_COLS_ZHENG].values
    n = len(legit)
    if n < 3: return None, None
    imp_pool = df[df["userid"]!=user][FEATURE_COLS_ZHENG].values
    if len(imp_pool) == 0: return None, None
    all_scores = []
    all_labels = []
    for i in range(n):
        x_train = np.delete(legit, i, axis=0)
        if len(x_train) < 2: continue
        x_legit_test = legit[i:i+1]
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

def evaluate_user(user, df39, dfzh, nu, gamma, mode):
    rng = np.random.RandomState(0)
    res = {}
    y39,s39   = (None,None)
    yzh,szh   = (None,None)
    if mode in ("features_only","fusion"):
        y39,s39 = pipeline_39_session(user, df39, rng, nu, gamma)
    if mode in ("zheng_only","fusion"):
        yzh,szh = pipeline_zheng_session(user, dfzh, rng, nu, gamma)
    if y39 is not None:
        res["features_39"] = compute_metrics(y39, s39)
    if yzh is not None:
        res["zheng"] = compute_metrics(yzh, szh)
    # true score-level fusion: both produce one score per session
    # align by taking the minimum number of scores across both
    if mode == "fusion" and y39 is not None and yzh is not None:
        # normalize both score vectors to [0,1] then average
        s39_norm  = normalize_scores(s39)
        szh_norm  = normalize_scores(szh)
        # match lengths — both use LOO so they should produce the same
        # number of legit scores (n sessions) but impostor counts may differ
        # separate legit and impostor, align, then recombine
        l39  = s39_norm[y39==1];  i39  = s39_norm[y39==0]
        lzh  = szh_norm[yzh==1];  izh  = szh_norm[yzh==0]
        nl   = min(len(l39), len(lzh))
        ni   = min(len(i39), len(izh))
        if nl > 0 and ni > 0:
            fused_l = (l39[:nl] + lzh[:nl]) / 2
            fused_i = (i39[:ni] + izh[:ni]) / 2
            y_fused = np.array([1]*nl + [0]*ni)
            s_fused = np.concatenate([fused_l, fused_i])
            res["fusion"] = compute_metrics(y_fused, s_fused)
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
    p.add_argument("--features",  required=True, help="39-feature CSV (with session column)")
    p.add_argument("--zheng",     required=True, help="Zheng 215-feature CSV")
    p.add_argument("--mode",      default="fusion",
                   choices=["fusion","features_only","zheng_only"])
    p.add_argument("--nu",        type=float, default=0.02)
    p.add_argument("--gamma",     type=float, default=0.01)
    p.add_argument("--users",     nargs="+", default=None)
    args = p.parse_args()
    df39 = pd.read_csv(args.features).replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURE_COLS_39)
    dfzh = pd.read_csv(args.zheng).replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURE_COLS_ZHENG)
    if "session" not in df39.columns:
        print("ERROR: features CSV has no session column.")
        print("Re-run: python extract_features.py --mode dataset ...")
        return
    users = args.users or sorted(df39["userid"].unique())
    r39,rzh,rfused = [],[],[]
    for u in users:
        print(f"Evaluating {u}...")
        r = evaluate_user(u,df39,dfzh,args.nu,args.gamma,args.mode)
        if "features_39" in r: r39.append((u,r["features_39"]))
        if "zheng"       in r: rzh.append((u,r["zheng"]))
        if "fusion"      in r: rfused.append((u,r["fusion"]))
    if r39:    print_table("39-Feature Pipeline (LOO session-averaged)", r39)
    if rzh:    print_table("Zheng 215-dim (LOO per session)", rzh)
    if rfused: print_table("Fusion: 39 Features + Zheng (true score fusion)", rfused)

if __name__ == "__main__":
    main()