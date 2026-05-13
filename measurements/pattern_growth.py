"""
pattern_growth.py

Implements the pattern-growth-based feature extraction from:
Chao Shen et al. "Continuous Authentication for Mouse Dynamics:
A Pattern-Growth Approach" (DSN 2012)

Pipeline:
  1. Load raw Balabit-format session files
  2. Segment into mouse operations (MM, PC, DD)
  3. Encode each operation as a tuple <action-type, screen-area>
     (application-type and window-position omitted — not in Balabit data)
  4. Mine frequent sequential behavior patterns using PrefixSpan
     with minimum support = 8%
  5. Extract 39 features only from actions that match a mined pattern
  6. Return feature DataFrame — same schema as extract_features.py output

Usage:
    python measurements/pattern_growth.py --data_dir balabit_dataset/training_files --out features/pattern_features.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from math import atan2, sqrt, pi
from collections import defaultdict
# PrefixSpan implemented directly to avoid memory issues with the
# third-party library on longer sequences
def _prefixspan(sequences, min_support, max_length=3):
    """
    Memory-efficient PrefixSpan sequential pattern mining.
    sequences: list of lists of hashable items
    min_support: minimum count (integer)
    max_length: maximum pattern length — caps combinatorial explosion
    Returns: list of (support_count, pattern) tuples
    """
    results = []

    def project(seqs, item):
        projected = []
        for seq in seqs:
            for i, s in enumerate(seq):
                if s == item:
                    projected.append(seq[i+1:])
                    break
        return projected

    def mine(seqs, prefix):
        if len(prefix) >= max_length:
            return
        counts = {}
        for seq in seqs:
            seen = set()
            for item in seq:
                if item not in seen:
                    counts[item] = counts.get(item, 0) + 1
                    seen.add(item)
        for item, count in counts.items():
            if count >= min_support:
                new_prefix = prefix + [item]
                results.append((count, new_prefix))
                mine(project(seqs, item), new_prefix)

    mine(sequences, [])
    return results

# ── Constants ──────────────────────────────────────────────────────────────────

COLUMNS       = ['record_timestamp', 'client_timestamp', 'button', 'state', 'x', 'y']
ACTION_MM     = 1   # mouse move
ACTION_PC     = 4   # point and click
ACTION_DD     = 3   # drag and drop
MIN_EVENTS    = 4
TIME_THRESHOLD = 10.0   # seconds — split MM actions on gaps larger than this
MIN_SUPPORT   = 0.5    # 8% as per Chao Shen paper

# Screen area encoding: divide screen into 3x3 = 9 regions
# We use percentile-based boundaries so they adapt to each user's screen
SCREEN_REGIONS = 3  # 3x3 grid → 9 regions (0-8)

BALABIT_USERS = ['user7', 'user9', 'user12', 'user15', 'user16',
                 'user20', 'user21', 'user23', 'user29', 'user35']

FEATURE_COLS = [
    'type_of_action', 'traveled_distance_pixel', 'elapsed_time',
    'direction_of_movement', 'straightness', 'num_points', 'sum_of_angles',
    'mean_curv', 'sd_curv', 'max_curv', 'min_curv',
    'mean_omega', 'sd_omega', 'max_omega', 'min_omega',
    'largest_deviation', 'dist_end_to_end_line', 'num_critical_points',
    'mean_vx', 'sd_vx', 'max_vx', 'min_vx',
    'mean_vy', 'sd_vy', 'max_vy', 'min_vy',
    'mean_v',  'sd_v',  'max_v',  'min_v',
    'mean_a',  'sd_a',  'max_a',  'min_a',
    'mean_jerk', 'sd_jerk', 'max_jerk', 'min_jerk',
    'a_beg_time', 'userid'
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
    """Load all session files for a user. Returns list of DataFrames."""
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
    """
    Segment raw events into MM, PC, DD actions.
    Returns list of dicts: {'type': int, 'events': DataFrame}
    """
    actions = []
    i = 0
    n = len(df)

    while i < n:
        state  = df.at[i, 'state']
        button = df.at[i, 'button']

        # Drag and Drop
        if state in ('Pressed', 'Down') and button in ('Left', 'left', 'Right', 'right'):
            j = i + 1
            while j < n and df.at[j, 'state'] not in ('Released', 'Up'):
                j += 1
            seg = df.iloc[i:j+1].reset_index(drop=True)
            if len(seg) >= MIN_EVENTS:
                actions.append({'type': ACTION_DD, 'events': seg})
            i = j + 1
            continue

        # Move — check if it ends in a click (PC) or is pure movement (MM)
        if state == 'Move':
            j = i
            while j < n and df.at[j, 'state'] == 'Move':
                if j > i and (df.at[j, 'client_timestamp'] -
                               df.at[j-1, 'client_timestamp']) > TIME_THRESHOLD:
                    break
                j += 1

            if j < n and df.at[j, 'state'] in ('Pressed', 'Down'):
                # Point and click
                k = j + 1
                while k < n and df.at[k, 'state'] not in ('Released', 'Up'):
                    k += 1
                seg = df.iloc[i:k+1].reset_index(drop=True)
                if len(seg) >= MIN_EVENTS:
                    actions.append({'type': ACTION_PC, 'events': seg})
                i = k + 1
            else:
                # Pure mouse move — split on time gaps
                mm = df.iloc[i:j].reset_index(drop=True)
                for sub in split_mm(mm):
                    actions.append({'type': ACTION_MM, 'events': sub})
                i = j
            continue

        i += 1

    return actions


def split_mm(df):
    """Split a move segment on time gaps > TIME_THRESHOLD."""
    segs = []
    start = 0
    for i in range(1, len(df)):
        if (df.at[i, 'client_timestamp'] -
                df.at[i-1, 'client_timestamp']) > TIME_THRESHOLD:
            sub = df.iloc[start:i].reset_index(drop=True)
            if len(sub) >= MIN_EVENTS:
                segs.append(sub)
            start = i
    sub = df.iloc[start:].reset_index(drop=True)
    if len(sub) >= MIN_EVENTS:
        segs.append(sub)
    return segs


# ── Screen area encoding ───────────────────────────────────────────────────────

def build_screen_encoder(all_sessions):
    """
    Compute x and y percentile boundaries across all sessions
    to define the 3x3 screen grid adaptively.
    """
    xs, ys = [], []
    for df in all_sessions:
        xs.extend(df['x'].tolist())
        ys.extend(df['y'].tolist())
    xs = np.array(xs)
    ys = np.array(ys)
    x_bounds = [np.percentile(xs, 33), np.percentile(xs, 66)]
    y_bounds = [np.percentile(ys, 33), np.percentile(ys, 66)]
    return x_bounds, y_bounds


def encode_screen_area(x, y, x_bounds, y_bounds):
    """Map (x, y) to a screen region 0-8 in a 3x3 grid."""
    col = 0 if x < x_bounds[0] else (1 if x < x_bounds[1] else 2)
    row = 0 if y < y_bounds[0] else (1 if y < y_bounds[1] else 2)
    return row * 3 + col  # 0-8


# ── Operation encoding ─────────────────────────────────────────────────────────

def encode_action(action, x_bounds, y_bounds):
    events = action['events']
    x_end = events.iloc[-1]['x']
    y_end = events.iloc[-1]['y']
    x_start = events.iloc[0]['x']
    y_start = events.iloc[0]['y']
    area      = encode_screen_area(x_end, y_end, x_bounds, y_bounds)
    direction = direction_8(x_end - x_start, y_end - y_start)  # 1-8
    return (action['type'], area, direction)


def encode_session(actions, x_bounds, y_bounds):
    """
    Encode only PC and DD actions as the sequence for pattern mining.
    MM actions are excluded — they are just cursor travel between
    intentional actions and carry no semantic pattern information.
    """
    encoded = []
    for a in actions:
        if a['type'] in (ACTION_PC, ACTION_DD):  # skip ACTION_MM
            encoded.append(encode_action(a, x_bounds, y_bounds))
    return encoded


# ── Pattern mining ─────────────────────────────────────────────────────────────

def mine_patterns(sequences, min_support_ratio):
    """
    Mine frequent sequential behavior patterns using PrefixSpan.

    sequences: list of lists of operation tuples (one list per session)
    min_support_ratio: float, e.g. 0.08 for 8%

    Returns: list of (support_count, pattern) tuples
    """
    n_sessions = len(sequences)
    min_support = max(1, int(min_support_ratio * n_sessions))

    # encode tuples as integers for speed
    vocab = {}
    int_sequences = []
    for seq in sequences:
        int_seq = []
        for op in seq:
            if op not in vocab:
                vocab[op] = len(vocab)
            int_seq.append(vocab[op])
        int_sequences.append(int_seq)

    # reverse vocab for decoding
    rev_vocab = {v: k for k, v in vocab.items()}

    raw_patterns = _prefixspan(int_sequences, min_support)

    # decode back to tuple patterns
    parsed = [(support, [rev_vocab[i] for i in pattern])
              for support, pattern in raw_patterns]

    print(f"  Mined {len(parsed)} frequent patterns "
          f"(min_support={min_support}/{n_sessions} sessions)")
    return parsed


def build_pattern_set(patterns, min_length=2):
    op_set = set()
    for _, pattern in patterns:
        if len(pattern) >= min_length:  # only multi-step patterns
            for op in pattern:
                op_set.add(op)
    return op_set


def action_matches_pattern(encoded_op, pattern_op_set):
    """Check if an encoded operation appears in any mined pattern."""
    return encoded_op in pattern_op_set


# ── Feature extraction (same as extract_features.py) ──────────────────────────

def compute_time_series(events):
    x = events['x'].values.astype(float)
    y = events['y'].values.astype(float)
    t = events['client_timestamp'].values.astype(float)
    n = len(x)

    vx    = np.zeros(n)
    vy    = np.zeros(n)
    v     = np.zeros(n)
    theta = np.zeros(n)
    a     = np.zeros(n)
    jerk  = np.zeros(n)
    omega = np.zeros(n)
    curv  = np.zeros(n)
    s     = np.zeros(n)

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

        dtheta = theta[i] - theta[i-1]
        while dtheta >  pi: dtheta -= 2*pi
        while dtheta < -pi: dtheta += 2*pi
        omega[i] = dtheta / dt
        curv[i]  = dtheta / max(ds, 1e-9)

    for i in range(1, n):
        dt = max(t[i] - t[i-1], 1e-9)
        a[i]    = (v[i] - v[i-1]) / dt
        jerk[i] = (a[i] - a[i-1]) / dt

    return dict(x=x, y=y, t=t, s=s,
                vx=vx, vy=vy, v=v,
                a=a, jerk=jerk, omega=omega,
                curvature=curv, theta=theta)


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

    traj_len  = s[-1]
    dx_total  = x[-1] - x[0]
    dy_total  = y[-1] - y[0]
    dist_e2e  = sqrt(dx_total**2 + dy_total**2)
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
        'mean_curv': mean_c, 'sd_curv': sd_c,
        'max_curv':  max_c,  'min_curv': min_c,
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
        'mean_jerk': mean_j, 'sd_jerk': sd_j,
        'max_jerk':  max_j,  'min_jerk': min_j,
        'a_beg_time': a_beg_time(a, t),
    }


# ── Per-user pattern-growth pipeline ──────────────────────────────────────────

def process_user(user_id, user_dir, min_support=MIN_SUPPORT):
    """
    Full pipeline for one user:
    1. Load sessions
    2. Segment into actions
    3. Build screen encoder
    4. Encode PC/DD operations only (MM excluded)
    5. Mine frequent patterns
    6. Extract features only from pattern-matching PC/DD actions
    """
    print(f"\n[{user_id}] Loading sessions...")
    sessions = load_user_sessions(user_dir)
    if not sessions:
        print(f"  No sessions found in {user_dir}")
        return pd.DataFrame()

    print(f"  {len(sessions)} sessions loaded")

    # build screen encoder from all sessions combined
    x_bounds, y_bounds = build_screen_encoder(sessions)

    # segment all sessions into actions and encode PC/DD only
    all_session_data = []   # list of (all_actions, pc_dd_actions, encoded_ops)
    total_actions = 0
    for df in sessions:
        all_actions = segment_actions(df)
        # filter to PC and DD only for pattern mining
        pc_dd_actions = [a for a in all_actions if a['type'] in (ACTION_PC, ACTION_DD)]
        encoded = [encode_action(a, x_bounds, y_bounds) for a in pc_dd_actions]
        all_session_data.append((all_actions, pc_dd_actions, encoded))
        total_actions += len(all_actions)

    total_pc_dd = sum(len(pc_dd) for _, pc_dd, _ in all_session_data)
    print(f"  {total_actions} total actions ({total_pc_dd} PC/DD)")

    # mine frequent patterns from PC/DD encoded sequences only
    print(f"  Mining patterns (min_support={min_support})...")
    op_sequences = [encoded for _, _, encoded in all_session_data]
    patterns = mine_patterns(op_sequences, min_support)

    if not patterns:
        print(f"  No patterns found — using all PC/DD actions")
        pattern_op_set = None
    else:
        pattern_op_set = build_pattern_set(patterns)
        print(f"  {len(pattern_op_set)} unique operations appear in patterns")

    # extract features only from PC/DD actions that match a pattern
    rows = []
    matched = 0
    for _, pc_dd_actions, encoded in all_session_data:
        for action, enc_op in zip(pc_dd_actions, encoded):
            if pattern_op_set is None or action_matches_pattern(enc_op, pattern_op_set):
                ts   = compute_time_series(action['events'])
                feat = extract_action_features(action['type'], ts)
                if feat is not None:
                    feat['userid'] = user_id
                    rows.append(feat)
                    matched += 1

    total_pc_dd_all = sum(len(pc_dd) for _, pc_dd, _ in all_session_data)
    pct = 100 * matched / total_pc_dd_all if total_pc_dd_all > 0 else 0
    print(f"  {matched}/{total_pc_dd_all} PC/DD actions matched patterns ({pct:.1f}%)")

    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Pattern-growth feature extraction (Chao Shen 2012)')
    parser.add_argument('--data_dir', default='training_files',
                        help='Root directory containing per-user subdirectories')
    parser.add_argument('--out', default='pattern_features.csv',
                        help='Output CSV path')
    parser.add_argument('--min_support', type=float, default=MIN_SUPPORT,
                        help='Minimum support ratio for pattern mining (default: 0.08)')
    parser.add_argument('--users', nargs='+', default=None,
                        help='Subset of users to process (default: all subdirs)')
    args = parser.parse_args()

    users = args.users or sorted(os.listdir(args.data_dir))
    all_dfs = []

    for user in users:
        user_dir = os.path.join(args.data_dir, user)
        if not os.path.isdir(user_dir):
            continue
        df = process_user(user, user_dir, min_support=args.min_support)
        if len(df) > 0:
            all_dfs.append(df)
            print(f"  → {len(df)} pattern-matched feature rows")

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        # drop inf/nan
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.dropna(subset=[c for c in result.columns if c != 'userid'])
        result.to_csv(args.out, index=False)
        print(f"\nSaved {len(result)} total rows to {args.out}")
    else:
        print("No data extracted.")


if __name__ == '__main__':
    main()