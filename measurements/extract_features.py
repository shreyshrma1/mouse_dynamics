"""
extract_features.py

Extracts the 39 mouse dynamics features from Balabit-format CSV files,
matching the feature set from Antal & Egyed-Zsigmond (2018) and the
existing balabit_39feat_PC_MM_DD_100.csv reference file.

Usage:
    # Extract from a directory of session files (like Balabit training_files)
    python extract_features.py --mode dataset --data_dir training_files --out features.csv

    # Extract from a single collected session file
    python extract_features.py --mode session --file session_1778450466.csv --user shrey --out my_features.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from math import atan2, sqrt, pi

# ── Constants ──────────────────────────────────────────────────────────────────

COLUMNS = ['record_timestamp', 'client_timestamp', 'button', 'state', 'x', 'y']

# Action type codes matching the reference CSV
# 1 = MM (mouse move), 3 = DD (drag and drop), 4 = PC (point and click)
ACTION_MM = 1
ACTION_PC = 4
ACTION_DD = 3

# Minimum number of events in an action (spline requires ≥ 4)
MIN_EVENTS = 4

# Time threshold for splitting MM actions (seconds)
TIME_THRESHOLD = 10.0

BALABIT_USERS = ['user7', 'user9', 'user12', 'user15', 'user16',
                 'user20', 'user21', 'user23', 'user29', 'user35']

# ── Data loading ───────────────────────────────────────────────────────────────

def load_session(filepath):
    """Load a single session CSV in Balabit format."""
    df = pd.read_csv(filepath, names=COLUMNS, header=None)
    # drop any header row that snuck in
    df = df[df['record_timestamp'] != 'record_timestamp']
    df['client_timestamp'] = pd.to_numeric(df['client_timestamp'], errors='coerce')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['client_timestamp', 'x', 'y'])
    df = df.reset_index(drop=True)
    return df


# ── Segmentation ───────────────────────────────────────────────────────────────

def segment_actions(df):
    """
    Segment raw events into MM, PC, and DD actions following Antal (2018).

    Returns a list of dicts: {'type': int, 'events': DataFrame}
    """
    actions = []
    i = 0
    n = len(df)

    while i < n:
        state = df.at[i, 'state']
        button = df.at[i, 'button']

        # ── Drag and Drop ──────────────────────────────────────────────────
        if state in ('Pressed', 'Down') and button in ('Left', 'left', 'Right', 'right'):
            # collect events until the matching Released/Up
            j = i + 1
            while j < n and df.at[j, 'state'] not in ('Released', 'Up'):
                j += 1
            segment = df.iloc[i:j+1].reset_index(drop=True)
            if len(segment) >= MIN_EVENTS:
                actions.append({'type': ACTION_DD, 'events': segment})
            i = j + 1
            continue

        # ── Point and Click ────────────────────────────────────────────────
        # MM movement followed by a click at the end
        if state == 'Move':
            # look ahead to see if this movement ends in a click
            j = i
            while j < n and df.at[j, 'state'] == 'Move':
                # split MM on large time gaps
                if j > i and (df.at[j, 'client_timestamp'] - df.at[j-1, 'client_timestamp']) > TIME_THRESHOLD:
                    break
                j += 1

            # check if the next event is a click
            if j < n and df.at[j, 'state'] in ('Pressed', 'Down'):
                # include up to the Released event
                k = j + 1
                while k < n and df.at[k, 'state'] not in ('Released', 'Up'):
                    k += 1
                segment = df.iloc[i:k+1].reset_index(drop=True)
                if len(segment) >= MIN_EVENTS:
                    actions.append({'type': ACTION_PC, 'events': segment})
                i = k + 1
            else:
                # pure MM segment — may need to split on time gaps
                mm_events = df.iloc[i:j].reset_index(drop=True)
                mm_actions = split_mm_on_gaps(mm_events)
                actions.extend(mm_actions)
                i = j
            continue

        # skip anything else (scroll, etc.)
        i += 1

    return actions


def split_mm_on_gaps(df):
    """Split a mouse-move segment into sub-segments on time gaps > TIME_THRESHOLD."""
    if len(df) == 0:
        return []
    actions = []
    start = 0
    for i in range(1, len(df)):
        dt = df.at[i, 'client_timestamp'] - df.at[i-1, 'client_timestamp']
        if dt > TIME_THRESHOLD:
            seg = df.iloc[start:i].reset_index(drop=True)
            if len(seg) >= MIN_EVENTS:
                actions.append({'type': ACTION_MM, 'events': seg})
            start = i
    seg = df.iloc[start:].reset_index(drop=True)
    if len(seg) >= MIN_EVENTS:
        actions.append({'type': ACTION_MM, 'events': seg})
    return actions


# ── Time series computation ────────────────────────────────────────────────────

def compute_time_series(events):
    """
    Compute the 7 time series from an action's events:
    vx, vy, v, a, jerk, omega, curvature
    Returns a dict of numpy arrays.
    """
    x = events['x'].values.astype(float)
    y = events['y'].values.astype(float)
    t = events['client_timestamp'].values.astype(float)
    n = len(x)

    # initialize
    vx = np.zeros(n)
    vy = np.zeros(n)
    v  = np.zeros(n)
    theta = np.zeros(n)  # angle of path tangent

    for i in range(1, n):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        dt = t[i] - t[i-1]
        if dt <= 0:
            dt = 1e-9
        vx[i] = dx / dt
        vy[i] = dy / dt
        v[i]  = sqrt(vx[i]**2 + vy[i]**2)
        theta[i] = atan2(dy, dx)

    # acceleration
    a = np.zeros(n)
    for i in range(1, n):
        dt = t[i] - t[i-1]
        if dt <= 0:
            dt = 1e-9
        a[i] = (v[i] - v[i-1]) / dt

    # jerk
    jerk = np.zeros(n)
    for i in range(1, n):
        dt = t[i] - t[i-1]
        if dt <= 0:
            dt = 1e-9
        jerk[i] = (a[i] - a[i-1]) / dt

    # angular velocity (omega)
    omega = np.zeros(n)
    for i in range(1, n):
        dt = t[i] - t[i-1]
        if dt <= 0:
            dt = 1e-9
        dtheta = theta[i] - theta[i-1]
        # wrap to [-pi, pi]
        while dtheta > pi:  dtheta -= 2*pi
        while dtheta < -pi: dtheta += 2*pi
        omega[i] = dtheta / dt

    # curvature: dtheta / ds
    curvature = np.zeros(n)
    s = np.zeros(n)
    for i in range(1, n):
        ds = sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
        s[i] = s[i-1] + ds
        ds = max(ds, 1e-9)
        dtheta = theta[i] - theta[i-1]
        while dtheta > pi:  dtheta -= 2*pi
        while dtheta < -pi: dtheta += 2*pi
        curvature[i] = dtheta / ds

    return {
        'x': x, 'y': y, 't': t, 's': s,
        'vx': vx, 'vy': vy, 'v': v,
        'a': a, 'jerk': jerk,
        'omega': omega, 'curvature': curvature,
        'theta': theta,
    }


# ── Individual feature computations ───────────────────────────────────────────

def direction_8(dx, dy):
    """Map end-to-end displacement to one of 8 directions (1-8)."""
    angle = atan2(dy, dx) * 180 / pi  # -180 to 180
    angle = angle % 360               # 0 to 360
    # 8 sectors of 45 degrees each, starting at 0 (East)
    sector = int((angle + 22.5) / 45) % 8 + 1
    return sector


def num_critical_points(theta):
    """Count direction reversals (sharp angles < 0.0005 rad threshold)."""
    count = 0
    for i in range(1, len(theta) - 1):
        dtheta = abs(theta[i] - theta[i-1])
        while dtheta > pi: dtheta -= 2*pi
        dtheta = abs(dtheta)
        if dtheta < 0.0005:
            count += 1
    return count


def a_beg_time(a, t):
    """Duration of the initial acceleration phase (while a > 0)."""
    for k in range(1, len(a)):
        if a[k] <= 0:
            return t[k] - t[0]
    return t[-1] - t[0]


# ── Per-action feature vector ──────────────────────────────────────────────────

def extract_action_features(action_type, ts):
    """
    Extract all 39 features from a single action's time series.
    Returns a dict matching the reference CSV column order.
    """
    x, y, t, s = ts['x'], ts['y'], ts['t'], ts['s']
    vx, vy, v  = ts['vx'], ts['vy'], ts['v']
    a, jerk    = ts['a'], ts['jerk']
    omega, c   = ts['omega'], ts['curvature']
    theta      = ts['theta']
    n = len(x)

    # skip degenerate actions
    if n < MIN_EVENTS:
        return None

    # trajectory length
    traj_len = s[-1]

    # end-to-end distance and direction
    dx_total = x[-1] - x[0]
    dy_total = y[-1] - y[0]
    dist_e2e = sqrt(dx_total**2 + dy_total**2)
    direction = direction_8(dx_total, dy_total)

    # straightness
    straightness = dist_e2e / traj_len if traj_len > 0 else 1.0

    # elapsed time
    elapsed = t[-1] - t[0]

    # sum of angles
    sum_angles = float(np.sum(theta))

    # largest deviation from end-to-end line
    if dist_e2e > 0:
        # distance from each point to the line from (x[0],y[0]) to (x[-1],y[-1])
        deviations = []
        for i in range(n):
            # cross product formula for point-to-line distance
            num = abs((y[-1]-y[0])*x[i] - (x[-1]-x[0])*y[i] + x[-1]*y[0] - y[-1]*x[0])
            deviations.append(num / dist_e2e)
        largest_dev = max(deviations)
    else:
        largest_dev = 0.0

    # num critical points
    n_crit = num_critical_points(theta)

    # acceleration beginning time
    a_beg = a_beg_time(a, t)

    def safe_stats(arr):
        """Return mean, std, max, min — ignoring first zero element."""
        arr = arr[1:] if len(arr) > 1 else arr
        if len(arr) == 0:
            return 0.0, 0.0, 0.0, 0.0
        return float(np.mean(arr)), float(np.std(arr)), float(np.max(arr)), float(np.min(arr))

    mean_vx, sd_vx, max_vx, min_vx = safe_stats(vx)
    mean_vy, sd_vy, max_vy, min_vy = safe_stats(vy)
    mean_v,  sd_v,  max_v,  min_v  = safe_stats(v)
    mean_a,  sd_a,  max_a,  min_a  = safe_stats(a)
    mean_j,  sd_j,  max_j,  min_j  = safe_stats(jerk)
    mean_om, sd_om, max_om, min_om = safe_stats(omega)
    mean_c,  sd_c,  max_c,  min_c  = safe_stats(c)

    return {
        'type_of_action':       action_type,
        'traveled_distance_pixel': traj_len,
        'elapsed_time':         elapsed,
        'direction_of_movement': direction,
        'straightness':         straightness,
        'num_points':           n,
        'sum_of_angles':        sum_angles,
        'mean_curv':            mean_c,
        'sd_curv':              sd_c,
        'max_curv':             max_c,
        'min_curv':             min_c,
        'mean_omega':           mean_om,
        'sd_omega':             sd_om,
        'max_omega':            max_om,
        'min_omega':            min_om,
        'largest_deviation':    largest_dev,
        'dist_end_to_end_line': dist_e2e,
        'num_critical_points':  n_crit,
        'mean_vx':              mean_vx,
        'sd_vx':                sd_vx,
        'max_vx':               max_vx,
        'min_vx':               min_vx,
        'mean_vy':              mean_vy,
        'sd_vy':                sd_vy,
        'max_vy':               max_vy,
        'min_vy':               min_vy,
        'mean_v':               mean_v,
        'sd_v':                 sd_v,
        'max_v':                max_v,
        'min_v':                min_v,
        'mean_a':               mean_a,
        'sd_a':                 sd_a,
        'max_a':                max_a,
        'min_a':                min_a,
        'mean_jerk':            mean_j,
        'sd_jerk':              sd_j,
        'max_jerk':             max_j,
        'min_jerk':             min_j,
        'a_beg_time':           a_beg,
    }


# ── Session-level extraction ───────────────────────────────────────────────────

def extract_session_features(filepath, user_id):
    """Extract features from a single session file. Returns a DataFrame."""
    df = load_session(filepath)
    actions = segment_actions(df)
    rows = []
    for action in actions:
        ts = compute_time_series(action['events'])
        feat = extract_action_features(action['type'], ts)
        if feat is not None:
            feat['userid'] = user_id
            rows.append(feat)
    return pd.DataFrame(rows)


def extract_user_features(user_dir, user_id):
    """Extract features from all sessions in a user directory."""
    all_rows = []
    for fname in sorted(os.listdir(user_dir)):
        fpath = os.path.join(user_dir, fname)
        df = extract_session_features(fpath, user_id)
        all_rows.append(df)
    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Extract 39 mouse dynamics features from Balabit-format data.')
    parser.add_argument('--mode', choices=['dataset', 'session'], required=True,
                        help='dataset: extract from a full Balabit-style directory; session: extract from a single file')
    parser.add_argument('--data_dir', default='training_files',
                        help='[dataset mode] root directory containing per-user subdirectories')
    parser.add_argument('--file', default=None,
                        help='[session mode] path to a single session CSV file')
    parser.add_argument('--user', default='unknown',
                        help='[session mode] user ID to assign to extracted features')
    parser.add_argument('--out', default='features_extracted.csv',
                        help='output CSV path')
    parser.add_argument('--users', nargs='+', default=None,
                        help='[dataset mode] subset of users to process (default: all subdirs)')
    args = parser.parse_args()

    if args.mode == 'dataset':
        data_dir = args.data_dir
        users = args.users or sorted(os.listdir(data_dir))
        all_dfs = []
        for user in users:
            user_dir = os.path.join(data_dir, user)
            if not os.path.isdir(user_dir):
                continue
            print(f"Processing {user}...")
            df = extract_user_features(user_dir, user)
            print(f"  → {len(df)} actions extracted")
            all_dfs.append(df)
        if all_dfs:
            result = pd.concat(all_dfs, ignore_index=True)
            result.to_csv(args.out, index=False)
            print(f"\nSaved {len(result)} total actions to {args.out}")
        else:
            print("No data found.")

    elif args.mode == 'session':
        if args.file is None:
            print("Error: --file required in session mode")
            return
        print(f"Processing {args.file}...")
        df = extract_session_features(args.file, args.user)
        df.to_csv(args.out, index=False)
        print(f"Saved {len(df)} actions to {args.out}")


if __name__ == '__main__':
    main()