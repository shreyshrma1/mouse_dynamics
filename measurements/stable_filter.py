"""
stable_filter.py

Alternative to pattern-growth mining for the Balabit dataset.
Instead of mining sequential patterns (which requires application context
not available in Balabit), this filters actions by how stable they are
within a user's own behavioral distribution.

The idea: extract features from ALL actions first, then keep only those
whose feature values fall within N standard deviations of the user's mean.
These stable, consistent actions are the equivalent of Chao Shen's
"frequent behavior segments" — the habitual, repeatable parts of behavior.

Pipeline:
  1. Load raw session files
  2. Segment into PC and DD actions only (MM excluded)
  3. Extract 39 features from all PC/DD actions
  4. For each user, compute per-feature mean and std across all actions
  5. Keep only actions within N std of the mean on key stability features
  6. Save filtered feature DataFrame

Usage:
    python measurements/stable_filter.py --data_dir balabit_dataset/training_files --out features/stable_features.csv
    python stable_filter.py --data_dir training_files --out stable_features.csv --n_std 1.0
"""

import os
import argparse
import numpy as np
import pandas as pd
from math import atan2, sqrt, pi

# ── Constants ──────────────────────────────────────────────────────────────────

COLUMNS        = ['record_timestamp', 'client_timestamp', 'button', 'state', 'x', 'y']
ACTION_MM      = 1
ACTION_PC      = 4
ACTION_DD      = 3
MIN_EVENTS     = 4
TIME_THRESHOLD = 10.0

# Features used to measure stability — these capture the core behavioral
# signature of a mouse action: how fast, how far, how long, how straight.
# Using too many features over-constrains the filter; these six capture
# the most stable and discriminative aspects per Table VIII of Antal (2018).
STABILITY_FEATURES = [
    'elapsed_time',
    'traveled_distance_pixel',
    'straightness',
    'mean_v',
    'mean_a',
    'a_beg_time',
]

# ── Data loading ───────────────────────────────────────────────────────────────

def load_session(filepath):
    df = pd.read_csv(filepath, names=COLUMNS, header=None)
    df = df[df['record_timestamp'] != 'record_timestamp']
    df['client_timestamp'] = pd.to_numeric(df['client_timestamp'], errors='coerce')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['client_timestamp', 'x', 'y']).reset_index(drop=True)
    return df


def load_user_sessions(user_dir):
    sessions = []
    for fname in sorted(os.listdir(user_dir)):
        fpath = os.path.join(user_dir, fname)
        try:
            df = load_session(fpath)
            if len(df) > 0:
                sessions.append(df)
        except Exception as e:
            print(f"  Warning: could not load {fpath}: {e}")
    return sessions


# ── Segmentation ───────────────────────────────────────────────────────────────

def segment_actions(df):
    actions = []
    i = 0
    n = len(df)

    while i < n:
        state  = df.at[i, 'state']
        button = df.at[i, 'button']

        if state in ('Pressed', 'Down') and button in ('Left', 'left', 'Right', 'right'):
            j = i + 1
            while j < n and df.at[j, 'state'] not in ('Released', 'Up'):
                j += 1
            seg = df.iloc[i:j+1].reset_index(drop=True)
            if len(seg) >= MIN_EVENTS:
                actions.append({'type': ACTION_DD, 'events': seg})
            i = j + 1
            continue

        if state == 'Move':
            j = i
            while j < n and df.at[j, 'state'] == 'Move':
                if j > i and (df.at[j, 'client_timestamp'] -
                               df.at[j-1, 'client_timestamp']) > TIME_THRESHOLD:
                    break
                j += 1

            if j < n and df.at[j, 'state'] in ('Pressed', 'Down'):
                k = j + 1
                while k < n and df.at[k, 'state'] not in ('Released', 'Up'):
                    k += 1
                seg = df.iloc[i:k+1].reset_index(drop=True)
                if len(seg) >= MIN_EVENTS:
                    actions.append({'type': ACTION_PC, 'events': seg})
                i = k + 1
            else:
                # skip pure MM actions entirely
                i = j
            continue

        i += 1

    return actions


# ── Feature extraction ─────────────────────────────────────────────────────────

def compute_time_series(events):
    x = events['x'].values.astype(float)
    y = events['y'].values.astype(float)
    t = events['client_timestamp'].values.astype(float)
    n = len(x)

    vx = vy = v = theta = a = jerk = omega = curv = s = np.zeros(n)
    vx, vy, v, theta = [np.zeros(n) for _ in range(4)]
    a, jerk, omega, curv, s = [np.zeros(n) for _ in range(5)]

    for i in range(1, n):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        dt = max(t[i] - t[i-1], 1e-9)
        ds = sqrt(dx**2 + dy**2)
        vx[i]    = dx / dt
        vy[i]    = dy / dt
        v[i]     = sqrt(vx[i]**2 + vy[i]**2)
        theta[i] = atan2(dy, dx)
        s[i]     = s[i-1] + ds
        dtheta   = theta[i] - theta[i-1]
        while dtheta >  pi: dtheta -= 2*pi
        while dtheta < -pi: dtheta += 2*pi
        omega[i] = dtheta / dt
        curv[i]  = dtheta / max(ds, 1e-9)

    for i in range(1, n):
        dt      = max(t[i] - t[i-1], 1e-9)
        a[i]    = (v[i] - v[i-1]) / dt
        jerk[i] = (a[i] - a[i-1]) / dt

    return dict(x=x, y=y, t=t, s=s, vx=vx, vy=vy, v=v,
                a=a, jerk=jerk, omega=omega, curvature=curv, theta=theta)


def direction_8(dx, dy):
    angle = (atan2(dy, dx) * 180 / pi) % 360
    return int((angle + 22.5) / 45) % 8 + 1


def num_critical_points(theta):
    count = 0
    for i in range(1, len(theta) - 1):
        dtheta = abs(theta[i] - theta[i-1])
        while dtheta > pi: dtheta -= 2*pi
        dtheta = abs(dtheta)
        if dtheta < 0.0005:
            count += 1
    return count


def a_beg_time(a, t):
    for k in range(1, len(a)):
        if a[k] <= 0:
            return t[k] - t[0]
    return t[-1] - t[0]


def safe_stats(arr):
    arr = arr[1:] if len(arr) > 1 else arr
    if len(arr) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr)), float(np.max(arr)), float(np.min(arr))


def extract_action_features(action_type, ts):
    x, y, t, s = ts['x'], ts['y'], ts['t'], ts['s']
    vx, vy, v  = ts['vx'], ts['vy'], ts['v']
    a, jerk    = ts['a'], ts['jerk']
    omega, c   = ts['omega'], ts['curvature']
    theta      = ts['theta']
    n = len(x)
    if n < MIN_EVENTS:
        return None

    traj_len = s[-1]
    dx_total = x[-1] - x[0]
    dy_total = y[-1] - y[0]
    dist_e2e = sqrt(dx_total**2 + dy_total**2)
    direction = direction_8(dx_total, dy_total)
    straight  = dist_e2e / traj_len if traj_len > 0 else 1.0
    elapsed   = t[-1] - t[0]

    if dist_e2e > 0:
        devs = [abs((y[-1]-y[0])*x[i] - (x[-1]-x[0])*y[i] +
                    x[-1]*y[0] - y[-1]*x[0]) / dist_e2e
                for i in range(n)]
        largest_dev = max(devs)
    else:
        largest_dev = 0.0

    mean_vx, sd_vx, max_vx, min_vx = safe_stats(vx)
    mean_vy, sd_vy, max_vy, min_vy = safe_stats(vy)
    mean_v,  sd_v,  max_v,  min_v  = safe_stats(v)
    mean_a,  sd_a,  max_a,  min_a  = safe_stats(a)
    mean_j,  sd_j,  max_j,  min_j  = safe_stats(jerk)
    mean_om, sd_om, max_om, min_om = safe_stats(omega)
    mean_c,  sd_c,  max_c,  min_c  = safe_stats(c)

    return {
        'type_of_action':          action_type,
        'traveled_distance_pixel': traj_len,
        'elapsed_time':            elapsed,
        'direction_of_movement':   direction,
        'straightness':            straight,
        'num_points':              n,
        'sum_of_angles':           float(np.sum(theta)),
        'mean_curv': mean_c,  'sd_curv': sd_c,
        'max_curv':  max_c,   'min_curv': min_c,
        'mean_omega': mean_om, 'sd_omega': sd_om,
        'max_omega':  max_om,  'min_omega': min_om,
        'largest_deviation':       largest_dev,
        'dist_end_to_end_line':    dist_e2e,
        'num_critical_points':     num_critical_points(theta),
        'mean_vx': mean_vx, 'sd_vx': sd_vx,
        'max_vx':  max_vx,  'min_vx': min_vx,
        'mean_vy': mean_vy, 'sd_vy': sd_vy,
        'max_vy':  max_vy,  'min_vy': min_vy,
        'mean_v':  mean_v,  'sd_v':  sd_v,
        'max_v':   max_v,   'min_v':  min_v,
        'mean_a':  mean_a,  'sd_a':  sd_a,
        'max_a':   max_a,   'min_a':  min_a,
        'mean_jerk': mean_j,  'sd_jerk': sd_j,
        'max_jerk':  max_j,   'min_jerk': min_j,
        'a_beg_time': a_beg_time(a, t),
    }


# ── Stability filtering ────────────────────────────────────────────────────────

def filter_stable_actions(df, n_std=1.5):
    """
    Keep only actions whose feature values fall within n_std standard
    deviations of the user's mean on each stability feature.

    This filters out outlier actions — unusual movements that don't
    reflect the user's habitual behavior — keeping only the stable,
    consistent part of their mouse dynamics.

    n_std controls how tight the filter is:
      1.0 → very strict, keeps ~68% of a normal distribution
      1.5 → moderate (default), keeps ~87%
      2.0 → loose, keeps ~95%
    """
    if len(df) == 0:
        return df

    mask = pd.Series([True] * len(df), index=df.index)

    for feat in STABILITY_FEATURES:
        if feat not in df.columns:
            continue
        col  = df[feat].replace([np.inf, -np.inf], np.nan).dropna()
        if len(col) == 0:
            continue
        mean = col.mean()
        std  = col.std()
        if std > 0:
            mask &= (df[feat] - mean).abs() <= n_std * std

    return df[mask]


# ── Per-user pipeline ──────────────────────────────────────────────────────────

def process_user(user_id, user_dir, n_std=1.5):
    print(f"\n[{user_id}] Loading sessions...")
    sessions = load_user_sessions(user_dir)
    if not sessions:
        print(f"  No sessions found.")
        return pd.DataFrame()

    print(f"  {len(sessions)} sessions loaded")

    # extract features from all PC/DD actions
    rows = []
    total_actions = 0
    for df in sessions:
        actions = segment_actions(df)
        pc_dd   = [a for a in actions if a['type'] in (ACTION_PC, ACTION_DD)]
        total_actions += len(pc_dd)
        for action in pc_dd:
            ts   = compute_time_series(action['events'])
            feat = extract_action_features(action['type'], ts)
            if feat is not None:
                feat['userid'] = user_id
                rows.append(feat)

    if not rows:
        print(f"  No features extracted.")
        return pd.DataFrame()

    all_features = pd.DataFrame(rows)
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    all_features = all_features.dropna(
        subset=[c for c in all_features.columns if c != 'userid']
    )

    print(f"  {len(all_features)} PC/DD actions extracted")

    # apply stability filter
    stable = filter_stable_actions(all_features, n_std=n_std)

    pct = 100 * len(stable) / len(all_features) if len(all_features) > 0 else 0
    print(f"  {len(stable)}/{len(all_features)} actions kept after "
          f"stability filter ({pct:.1f}%) at n_std={n_std}")

    return stable


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Stable-action feature extraction for mouse dynamics OCSVM.')
    parser.add_argument('--data_dir', default='training_files',
                        help='Root directory with per-user subdirectories')
    parser.add_argument('--out', default='stable_features.csv',
                        help='Output CSV path')
    parser.add_argument('--n_std', type=float, default=1.5,
                        help='Standard deviation threshold for stability filter '
                             '(default: 1.5 — keeps actions within 1.5 std of mean)')
    parser.add_argument('--users', nargs='+', default=None,
                        help='Subset of users to process (default: all subdirs)')
    args = parser.parse_args()

    users   = args.users or sorted(os.listdir(args.data_dir))
    all_dfs = []

    for user in users:
        user_dir = os.path.join(args.data_dir, user)
        if not os.path.isdir(user_dir):
            continue
        df = process_user(user, user_dir, n_std=args.n_std)
        if len(df) > 0:
            all_dfs.append(df)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        result.to_csv(args.out, index=False)
        print(f"\nSaved {len(result)} total rows to {args.out}")
        print(f"Users: {result['userid'].value_counts().to_dict()}")
    else:
        print("No data extracted.")


if __name__ == '__main__':
    main()