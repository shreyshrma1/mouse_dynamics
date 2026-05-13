"""
extract_zheng_features.py

Implements the angle-based feature extraction from:
Zheng, Paloski, and Wang. "An Efficient User Verification System
Using Angle-Based Mouse Movement Biometrics." ACM CCS 2011 / ACM TISS 2016.

Produces one row per SESSION (not per action) — a 215-dimensional histogram
vector: 180-bin curvature angle distribution + 35-bin curvature distance
distribution.

Usage:
    python extract_zheng_features.py --mode dataset --data_dir training_files --out zheng_features.csv
    python extract_zheng_features.py --mode session --file session.csv --user shrey --out zheng_session.csv
"""

import os
import argparse
import math
import numpy as np
import pandas as pd
from math import sqrt

COLUMNS        = ['record_timestamp', 'client_timestamp', 'button', 'state', 'x', 'y']
MIN_EVENTS     = 4
TIME_THRESHOLD = 60.0
ANGLE_BINS     = 180
DIST_BINS      = 35
DIST_MAX       = 0.35
ANGLE_COLS     = [f'ca_{i}' for i in range(ANGLE_BINS)]
DIST_COLS      = [f'cd_{i}' for i in range(DIST_BINS)]


def load_session(filepath):
    df = pd.read_csv(filepath, names=COLUMNS, header=None)
    df = df[df['record_timestamp'] != 'record_timestamp']
    df['client_timestamp'] = pd.to_numeric(df['client_timestamp'], errors='coerce')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    return df.dropna(subset=['client_timestamp', 'x', 'y']).reset_index(drop=True)


def load_user_sessions(user_dir):
    sessions, fnames = [], []
    for fname in sorted(os.listdir(user_dir)):
        fpath = os.path.join(user_dir, fname)
        try:
            df = load_session(fpath)
            if len(df) > 0:
                sessions.append(df)
                fnames.append(fname)
        except Exception as e:
            print(f"  Warning: {fpath}: {e}")
    return sessions, fnames


def segment_pc_actions(df):
    """Return list of event DataFrames for PC and DD actions only."""
    actions = []
    i, n = 0, len(df)
    while i < n:
        state  = df.at[i, 'state']
        button = df.at[i, 'button']

        if state in ('Pressed', 'Down') and button in ('Left', 'left', 'Right', 'right'):
            j = i + 1
            while j < n and df.at[j, 'state'] not in ('Released', 'Up'):
                j += 1
            seg = df.iloc[i:j+1].reset_index(drop=True)
            if len(seg) >= MIN_EVENTS:
                actions.append(seg)
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
                    actions.append(seg)
                i = k + 1
            else:
                i = j
            continue

        i += 1
    return actions


def compute_curvature_angles(events):
    """
    Angle at each middle point of a triplet (A, B, C) — the angle
    between vectors BA and BC, in degrees [0, 180].
    180 = straight movement, 0 = sharp reversal.
    """
    x = events['x'].values.astype(float)
    y = events['y'].values.astype(float)
    angles = []
    for i in range(1, len(x) - 1):
        ax, ay = x[i-1] - x[i], y[i-1] - y[i]
        bx, by = x[i+1] - x[i], y[i+1] - y[i]
        mag_a = sqrt(ax**2 + ay**2)
        mag_b = sqrt(bx**2 + by**2)
        if mag_a < 1e-9 or mag_b < 1e-9:
            continue
        cos_t = max(-1.0, min(1.0, (ax*bx + ay*by) / (mag_a * mag_b)))
        angles.append(math.degrees(math.acos(cos_t)))
    return np.array(angles)


def compute_curvature_distances(events):
    """
    Arc lengths between consecutive inflection points (angle < 90 deg),
    normalized by total trajectory length so they're resolution-independent.
    """
    x = events['x'].values.astype(float)
    y = events['y'].values.astype(float)
    n = len(x)

    ds = np.array([0.0] + [sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
                            for i in range(1, n)])
    total = ds.sum()
    if total < 1e-9:
        return np.array([])

    inflections = [0]
    for i in range(1, n - 1):
        ax, ay = x[i-1] - x[i], y[i-1] - y[i]
        bx, by = x[i+1] - x[i], y[i+1] - y[i]
        ma = sqrt(ax**2 + ay**2)
        mb = sqrt(bx**2 + by**2)
        if ma < 1e-9 or mb < 1e-9:
            continue
        cos_t = max(-1.0, min(1.0, (ax*bx + ay*by) / (ma * mb)))
        if math.degrees(math.acos(cos_t)) < 90:
            inflections.append(i)
    inflections.append(n - 1)

    cumulative = np.cumsum(ds)
    distances = []
    for k in range(1, len(inflections)):
        arc = cumulative[inflections[k]] - cumulative[inflections[k-1]]
        distances.append(arc / total)

    return np.array(distances)


def extract_zheng_features_session(session_df):
    """
    Pool curvature angles and distances across all PC/DD actions in a session,
    then build normalized 180-bin and 35-bin histograms.
    Returns (angle_hist, dist_hist) or (None, None) if no actions found.
    """
    actions = segment_pc_actions(session_df)
    all_angles, all_distances = [], []

    for events in actions:
        all_angles.extend(compute_curvature_angles(events).tolist())
        all_distances.extend(compute_curvature_distances(events).tolist())

    if not all_angles:
        return None, None

    angle_hist, _ = np.histogram(
        np.array(all_angles), bins=ANGLE_BINS, range=(0, 180), density=True)
    dist_hist, _  = np.histogram(
        np.clip(np.array(all_distances), 0, DIST_MAX),
        bins=DIST_BINS, range=(0, DIST_MAX), density=True)

    return np.nan_to_num(angle_hist), np.nan_to_num(dist_hist)


def features_to_row(angle_hist, dist_hist, user_id, session_name):
    row = {}
    for i, v in enumerate(angle_hist):
        row[f'ca_{i}'] = v
    for i, v in enumerate(dist_hist):
        row[f'cd_{i}'] = v
    row['userid']  = user_id
    row['session'] = session_name
    return row


def extract_user_zheng_features(user_dir, user_id, window_size=None):
    """
    Extract Zheng features for all sessions of a user.

    If window_size is set, each session file is sliced into windows of
    that many PC/DD actions, producing one histogram row per window.
    Window labels match extract_features.py: filename_wN.
    """
    sessions, fnames = load_user_sessions(user_dir)
    rows = []

    for session_df, fname in zip(sessions, fnames):
        actions = segment_pc_actions(session_df)

        if window_size is None:
            # one histogram per file
            angle_hist, dist_hist = extract_zheng_features_session(session_df)
            if angle_hist is None:
                print(f"  Warning: no PC actions in {fname}, skipping")
                continue
            rows.append(features_to_row(angle_hist, dist_hist, user_id, fname))
        else:
            # one histogram per window of window_size PC/DD actions
            for w_idx, start in enumerate(range(0, len(actions), window_size)):
                window = actions[start:start + window_size]
                if len(window) < window_size // 2:
                    continue
                all_angles, all_distances = [], []
                for events in window:
                    all_angles.extend(compute_curvature_angles(events).tolist())
                    all_distances.extend(compute_curvature_distances(events).tolist())
                if not all_angles:
                    continue
                angle_hist, _ = np.histogram(
                    np.array(all_angles), bins=ANGLE_BINS,
                    range=(0, 180), density=True)
                dist_hist, _ = np.histogram(
                    np.clip(np.array(all_distances), 0, DIST_MAX),
                    bins=DIST_BINS, range=(0, DIST_MAX), density=True)
                angle_hist = np.nan_to_num(angle_hist)
                dist_hist  = np.nan_to_num(dist_hist)
                rows.append(features_to_row(
                    angle_hist, dist_hist, user_id, f'{fname}_w{w_idx}'))

    return pd.DataFrame(rows)


def extract_session_zheng_features(filepath, user_id):
    df = load_session(filepath)
    angle_hist, dist_hist = extract_zheng_features_session(df)
    if angle_hist is None:
        print("Warning: no PC actions found")
        return pd.DataFrame()
    return pd.DataFrame([features_to_row(
        angle_hist, dist_hist, user_id, os.path.basename(filepath))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',        choices=['dataset', 'session'], required=True)
    parser.add_argument('--data_dir',    default='training_files')
    parser.add_argument('--file',        default=None)
    parser.add_argument('--user',        default='unknown')
    parser.add_argument('--out',         default='zheng_features.csv')
    parser.add_argument('--users',       nargs='+', default=None)
    parser.add_argument('--window_size', type=int, default=None,
                        help='Slice each session into windows of this many PC/DD actions '
                             '(e.g. 200). Must match --window_size used in extract_features.py.')
    args = parser.parse_args()

    if args.mode == 'dataset':
        users   = args.users or sorted(os.listdir(args.data_dir))
        all_dfs = []
        for user in users:
            user_dir = os.path.join(args.data_dir, user)
            if not os.path.isdir(user_dir):
                continue
            print(f"Processing {user}...")
            df = extract_user_zheng_features(user_dir, user, window_size=args.window_size)
            print(f"  -> {len(df)} windows")
            all_dfs.append(df)
        if all_dfs:
            result = pd.concat(all_dfs, ignore_index=True)
            result.to_csv(args.out, index=False)
            print(f"\nSaved {len(result)} rows to {args.out}")
        else:
            print("No data found.")

    elif args.mode == 'session':
        if args.file is None:
            print("Error: --file required in session mode")
            return
        df = extract_session_zheng_features(args.file, args.user)
        df.to_csv(args.out, index=False)
        print(f"Saved {len(df)} rows to {args.out}")


if __name__ == '__main__':
    main()