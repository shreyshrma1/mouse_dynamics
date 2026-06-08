"""
extract_features_scroll.py

Identical to extract_features.py, but extract_session_features() also
computes per-window scroll features and attaches them to every row in
that window.

Scroll features (7 + 3 burst = 10 additional columns):
    scroll_count          - number of scroll events in the window
    scroll_rate           - scrolls per second
    scroll_ratio          - scroll events / total raw events
    scroll_up_ratio       - fraction of scrolls that are upward
    scroll_dur_mean       - mean duration of individual scroll events (inter-scroll gap)
    scroll_dur_std        - std  of inter-scroll gaps
    scroll_burst_count    - number of bursts (gap > BURST_THRESHOLD_S apart)
    scroll_burst_dur_mean - mean burst duration (ms)
    scroll_burst_len_mean - mean number of events per burst

Optional --more_scroll features (deltaY-based, collected data only):
    scroll_dy_abs_mean, scroll_dy_abs_std, scroll_dy_abs_max,
    scroll_dy_total, scroll_dy_up_ratio, scroll_speed_mean,
    scroll_speed_std, scroll_accel_mean, scroll_accel_std,
    scroll_burst_dy_mean, scroll_burst_dy_per_tick, scroll_speed_intensity

Optional --dir_scroll features (directional burst split):
    scroll_up_count, scroll_down_count,
    scroll_up_burst_count, scroll_down_burst_count,
    scroll_up_burst_dur_mean, scroll_down_burst_dur_mean,
    scroll_up_burst_len_mean, scroll_down_burst_len_mean

Rows that come from windows with zero scroll events get 0 for all
scroll columns (scroll_up_ratio is also 0 in that case).
"""

import os
import argparse
import numpy as np
import pandas as pd
from math import atan2, sqrt, pi

# ── Constants ──────────────────────────────────────────────────────────────────

COLUMNS = ['record_timestamp', 'client_timestamp', 'button', 'state', 'x', 'y']

ACTION_MM = 1
ACTION_PC = 4
ACTION_DD = 3

MIN_EVENTS = 4
TIME_THRESHOLD = 60.0

# Gap in seconds that separates two scroll bursts
BURST_THRESHOLD_S = 0.5

BALABIT_USERS = ['user7', 'user9', 'user12', 'user15', 'user16',
                 'user20', 'user21', 'user23', 'user29', 'user35']

SCROLL_COLS = [
    'scroll_count',
    'scroll_rate',
    'scroll_ratio',
    'scroll_up_ratio',
    'scroll_dur_mean',
    'scroll_dur_std',
    'scroll_burst_count',
    'scroll_burst_dur_mean',
    'scroll_burst_len_mean',
]

# Additional scroll features requiring deltaY (collected data only, --more_scroll)
MORE_SCROLL_COLS = [
    'scroll_dy_abs_mean',
    'scroll_dy_abs_std',
    'scroll_dy_abs_max',
    'scroll_dy_total',
    'scroll_dy_up_ratio',
    'scroll_speed_mean',
    'scroll_speed_std',
    'scroll_accel_mean',
    'scroll_accel_std',
    'scroll_burst_dy_mean',
    'scroll_burst_dy_per_tick',
    'scroll_speed_intensity',
]

# Directional burst features (--dir_scroll)
DIR_SCROLL_COLS = [
    'scroll_up_count',
    'scroll_down_count',
    'scroll_up_burst_count',
    'scroll_down_burst_count',
    'scroll_up_burst_dur_mean',
    'scroll_down_burst_dur_mean',
    'scroll_up_burst_len_mean',
    'scroll_down_burst_len_mean',
]

# ── Data loading ───────────────────────────────────────────────────────────────

def load_session(filepath):
    df = pd.read_csv(filepath, names=COLUMNS, header=None)
    df = df[df['record_timestamp'] != 'record_timestamp']
    df['client_timestamp'] = pd.to_numeric(df['client_timestamp'], errors='coerce')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['client_timestamp', 'x', 'y'])
    df = df.reset_index(drop=True)
    return df


# ── Scroll feature extraction ──────────────────────────────────────────────────

def _dir_bursts(ts_arr):
    """Compute burst count, mean duration, mean length for a directional subset."""
    if len(ts_arr) == 0:
        return 0.0, 0.0, 0.0
    if len(ts_arr) == 1:
        return 1.0, 0.0, 1.0
    bursts = []
    burst_start = 0
    for i in range(1, len(ts_arr)):
        if ts_arr[i] - ts_arr[i - 1] > BURST_THRESHOLD_S:
            bursts.append((burst_start, i))
            burst_start = i
    bursts.append((burst_start, len(ts_arr)))
    durs = [ts_arr[e - 1] - ts_arr[s] for s, e in bursts]
    lens = [e - s for s, e in bursts]
    return float(len(bursts)), float(np.mean(durs)), float(np.mean(lens))


def extract_scroll_features(raw_df, window_start_idx, window_end_idx,
                             more_scroll=False, dir_scroll=False):
    """
    Compute scroll features from the raw event rows that correspond to
    a given window (indexed by raw_df row positions).

    raw_df               - full session DataFrame (all raw events)
    window_start_idx / window_end_idx - row slice of the window in raw_df
    more_scroll          - if True, also compute deltaY-based features
    dir_scroll           - if True, also compute directional burst features

    Returns a dict with keys matching SCROLL_COLS
    (+ MORE_SCROLL_COLS if more_scroll, + DIR_SCROLL_COLS if dir_scroll).
    """
    all_cols = (SCROLL_COLS
                + (MORE_SCROLL_COLS if more_scroll else [])
                + (DIR_SCROLL_COLS  if dir_scroll  else []))
    zero = {c: 0.0 for c in all_cols}

    window_raw = raw_df.iloc[window_start_idx:window_end_idx]
    total_events = len(window_raw)

    scrolls = window_raw[window_raw['button'] == 'Scroll'].copy()
    scroll_count = len(scrolls)

    if scroll_count == 0 or total_events == 0:
        return zero

    # ── Rate and ratio ─────────────────────────────────────────────────────
    window_duration = (window_raw['client_timestamp'].iloc[-1]
                       - window_raw['client_timestamp'].iloc[0])
    scroll_rate  = scroll_count / window_duration if window_duration > 0 else 0.0
    scroll_ratio = scroll_count / total_events

    # ── Direction (up/down by event count) ────────────────────────────────
    up_count = (scrolls['state'] == 'Up').sum()
    scroll_up_ratio = up_count / scroll_count

    # ── Inter-scroll intervals ─────────────────────────────────────────────
    ts = scrolls['client_timestamp'].values
    if scroll_count >= 2:
        gaps = np.diff(ts)
        scroll_dur_mean = float(np.mean(gaps))
        scroll_dur_std  = float(np.std(gaps))
    else:
        gaps = np.array([])
        scroll_dur_mean = 0.0
        scroll_dur_std  = 0.0

    # ── Bursts ─────────────────────────────────────────────────────────────
    if scroll_count >= 2:
        bursts = []
        burst_start = 0
        for i in range(1, len(ts)):
            if ts[i] - ts[i - 1] > BURST_THRESHOLD_S:
                bursts.append((burst_start, i))
                burst_start = i
        bursts.append((burst_start, len(ts)))

        burst_durations = [ts[e - 1] - ts[s] for s, e in bursts]
        burst_lengths   = [e - s for s, e in bursts]

        scroll_burst_count    = float(len(bursts))
        scroll_burst_dur_mean = float(np.mean(burst_durations))
        scroll_burst_len_mean = float(np.mean(burst_lengths))
    else:
        bursts = [(0, scroll_count)]
        burst_durations = [0.0]
        burst_lengths   = [scroll_count]
        scroll_burst_count    = 1.0 if scroll_count == 1 else 0.0
        scroll_burst_dur_mean = 0.0
        scroll_burst_len_mean = float(scroll_count)

    result = {
        'scroll_count':          float(scroll_count),
        'scroll_rate':           scroll_rate,
        'scroll_ratio':          scroll_ratio,
        'scroll_up_ratio':       scroll_up_ratio,
        'scroll_dur_mean':       scroll_dur_mean,
        'scroll_dur_std':        scroll_dur_std,
        'scroll_burst_count':    scroll_burst_count,
        'scroll_burst_dur_mean': scroll_burst_dur_mean,
        'scroll_burst_len_mean': scroll_burst_len_mean,
    }

    # ── deltaY features (--more_scroll only) ──────────────────────────────
    if more_scroll:
        dy = scrolls['y'].values.astype(float)
        dy_abs = np.abs(dy)

        scroll_dy_abs_mean = float(np.mean(dy_abs))
        scroll_dy_abs_std  = float(np.std(dy_abs))
        scroll_dy_abs_max  = float(np.max(dy_abs))
        scroll_dy_total    = float(np.sum(dy_abs))

        up_dy    = float(np.sum(dy_abs[dy > 0]))
        down_dy  = float(np.sum(dy_abs[dy < 0]))
        total_dy = up_dy + down_dy
        scroll_dy_up_ratio = up_dy / total_dy if total_dy > 0 else 0.0

        if scroll_count >= 2:
            speeds = dy_abs[1:] / np.where(gaps > 0, gaps, 1e-9)
            scroll_speed_mean = float(np.mean(speeds))
            scroll_speed_std  = float(np.std(speeds))
            if len(speeds) >= 2:
                accels = np.diff(speeds)
                scroll_accel_mean = float(np.mean(accels))
                scroll_accel_std  = float(np.std(accels))
            else:
                scroll_accel_mean = 0.0
                scroll_accel_std  = 0.0
            scroll_speed_intensity = scroll_speed_mean
        else:
            scroll_speed_mean      = 0.0
            scroll_speed_std       = 0.0
            scroll_accel_mean      = 0.0
            scroll_accel_std       = 0.0
            scroll_speed_intensity = 0.0

        burst_dy_totals   = [float(np.sum(np.abs(dy[s:e]))) for s, e in bursts]
        burst_dy_per_tick = [float(np.mean(np.abs(dy[s:e]))) if e > s else 0.0
                             for s, e in bursts]
        scroll_burst_dy_mean     = float(np.mean(burst_dy_totals))
        scroll_burst_dy_per_tick = float(np.mean(burst_dy_per_tick))

        result.update({
            'scroll_dy_abs_mean':       scroll_dy_abs_mean,
            'scroll_dy_abs_std':        scroll_dy_abs_std,
            'scroll_dy_abs_max':        scroll_dy_abs_max,
            'scroll_dy_total':          scroll_dy_total,
            'scroll_dy_up_ratio':       scroll_dy_up_ratio,
            'scroll_speed_mean':        scroll_speed_mean,
            'scroll_speed_std':         scroll_speed_std,
            'scroll_accel_mean':        scroll_accel_mean,
            'scroll_accel_std':         scroll_accel_std,
            'scroll_burst_dy_mean':     scroll_burst_dy_mean,
            'scroll_burst_dy_per_tick': scroll_burst_dy_per_tick,
            'scroll_speed_intensity':   scroll_speed_intensity,
        })

    # ── Directional burst features (--dir_scroll only) ────────────────────
    if dir_scroll:
        up_scrolls   = scrolls[scrolls['state'] == 'Up']
        down_scrolls = scrolls[scrolls['state'] == 'Down']

        up_ts   = up_scrolls['client_timestamp'].values
        down_ts = down_scrolls['client_timestamp'].values

        up_bc,   up_bdur,   up_blen   = _dir_bursts(up_ts)
        down_bc, down_bdur, down_blen = _dir_bursts(down_ts)

        result.update({
            'scroll_up_count':            float(len(up_scrolls)),
            'scroll_down_count':          float(len(down_scrolls)),
            'scroll_up_burst_count':      up_bc,
            'scroll_down_burst_count':    down_bc,
            'scroll_up_burst_dur_mean':   up_bdur,
            'scroll_down_burst_dur_mean': down_bdur,
            'scroll_up_burst_len_mean':   up_blen,
            'scroll_down_burst_len_mean': down_blen,
        })

    return result


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
            segment = df.iloc[i:j+1].reset_index(drop=True)
            if len(segment) >= MIN_EVENTS:
                actions.append({'type': ACTION_DD, 'events': segment,
                                'raw_start': i, 'raw_end': j + 1})
            i = j + 1
            continue

        if state == 'Move':
            j = i
            while j < n and df.at[j, 'state'] == 'Move':
                if j > i and (df.at[j, 'client_timestamp'] - df.at[j-1, 'client_timestamp']) > TIME_THRESHOLD:
                    break
                j += 1

            if j < n and df.at[j, 'state'] in ('Pressed', 'Down'):
                k = j + 1
                while k < n and df.at[k, 'state'] not in ('Released', 'Up'):
                    k += 1
                segment = df.iloc[i:k+1].reset_index(drop=True)
                if len(segment) >= MIN_EVENTS:
                    actions.append({'type': ACTION_PC, 'events': segment,
                                    'raw_start': i, 'raw_end': k + 1})
                i = k + 1
            else:
                mm_events = df.iloc[i:j].reset_index(drop=True)
                mm_actions = split_mm_on_gaps(mm_events, raw_offset=i)
                actions.extend(mm_actions)
                i = j
            continue

        i += 1

    return actions


def split_mm_on_gaps(df, raw_offset=0):
    if len(df) == 0:
        return []
    actions = []
    start = 0
    for i in range(1, len(df)):
        dt = df.at[i, 'client_timestamp'] - df.at[i-1, 'client_timestamp']
        if dt > TIME_THRESHOLD:
            seg = df.iloc[start:i].reset_index(drop=True)
            if len(seg) >= MIN_EVENTS:
                actions.append({'type': ACTION_MM, 'events': seg,
                                'raw_start': raw_offset + start,
                                'raw_end':   raw_offset + i})
            start = i
    seg = df.iloc[start:].reset_index(drop=True)
    if len(seg) >= MIN_EVENTS:
        actions.append({'type': ACTION_MM, 'events': seg,
                        'raw_start': raw_offset + start,
                        'raw_end':   raw_offset + len(df)})
    return actions


# ── Time series computation ────────────────────────────────────────────────────

def compute_time_series(events):
    x = events['x'].values.astype(float)
    y = events['y'].values.astype(float)
    t = events['client_timestamp'].values.astype(float)
    n = len(x)

    vx = np.zeros(n); vy = np.zeros(n); v = np.zeros(n)
    theta = np.zeros(n)

    for i in range(1, n):
        dx = x[i] - x[i-1]; dy = y[i] - y[i-1]
        dt = t[i] - t[i-1]
        if dt <= 0: dt = 1e-9
        vx[i] = dx/dt; vy[i] = dy/dt
        v[i]  = sqrt(vx[i]**2 + vy[i]**2)
        theta[i] = atan2(dy, dx)

    a = np.zeros(n)
    for i in range(1, n):
        dt = t[i] - t[i-1]
        if dt <= 0: dt = 1e-9
        a[i] = (v[i] - v[i-1]) / dt

    jerk = np.zeros(n)
    for i in range(1, n):
        dt = t[i] - t[i-1]
        if dt <= 0: dt = 1e-9
        jerk[i] = (a[i] - a[i-1]) / dt

    omega = np.zeros(n)
    for i in range(1, n):
        dt = t[i] - t[i-1]
        if dt <= 0: dt = 1e-9
        dtheta = theta[i] - theta[i-1]
        while dtheta > pi:  dtheta -= 2*pi
        while dtheta < -pi: dtheta += 2*pi
        omega[i] = dtheta / dt

    curvature = np.zeros(n); s = np.zeros(n)
    for i in range(1, n):
        ds = sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
        s[i] = s[i-1] + ds
        ds = max(ds, 1e-9)
        dtheta = theta[i] - theta[i-1]
        while dtheta > pi:  dtheta -= 2*pi
        while dtheta < -pi: dtheta += 2*pi
        curvature[i] = dtheta / ds

    return {'x': x, 'y': y, 't': t, 's': s,
            'vx': vx, 'vy': vy, 'v': v,
            'a': a, 'jerk': jerk,
            'omega': omega, 'curvature': curvature,
            'theta': theta}


# ── Individual feature computations ───────────────────────────────────────────

def direction_8(dx, dy):
    angle = atan2(dy, dx) * 180 / pi
    angle = angle % 360
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


# ── Per-action feature vector ──────────────────────────────────────────────────

def extract_action_features(action_type, ts):
    x, y, t, s = ts['x'], ts['y'], ts['t'], ts['s']
    vx, vy, v  = ts['vx'], ts['vy'], ts['v']
    a, jerk    = ts['a'], ts['jerk']
    omega, c   = ts['omega'], ts['curvature']
    theta      = ts['theta']
    n = len(x)

    if n < MIN_EVENTS:
        return None

    traj_len    = s[-1]
    dx_total    = x[-1] - x[0]; dy_total = y[-1] - y[0]
    dist_e2e    = sqrt(dx_total**2 + dy_total**2)
    direction   = direction_8(dx_total, dy_total)
    straightness = dist_e2e / traj_len if traj_len > 0 else 1.0
    elapsed     = t[-1] - t[0]
    sum_angles  = float(np.sum(theta))

    if dist_e2e > 0:
        deviations = []
        for i in range(n):
            num = abs((y[-1]-y[0])*x[i] - (x[-1]-x[0])*y[i] + x[-1]*y[0] - y[-1]*x[0])
            deviations.append(num / dist_e2e)
        largest_dev = max(deviations)
    else:
        largest_dev = 0.0

    n_crit = num_critical_points(theta)
    a_beg  = a_beg_time(a, t)

    def safe_stats(arr):
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
        'type_of_action':          action_type,
        'traveled_distance_pixel': traj_len,
        'elapsed_time':            elapsed,
        'direction_of_movement':   direction,
        'straightness':            straightness,
        'num_points':              n,
        'sum_of_angles':           sum_angles,
        'mean_curv':               mean_c,  'sd_curv':  sd_c,
        'max_curv':                max_c,   'min_curv': min_c,
        'mean_omega':              mean_om, 'sd_omega': sd_om,
        'max_omega':               max_om,  'min_omega': min_om,
        'largest_deviation':       largest_dev,
        'dist_end_to_end_line':    dist_e2e,
        'num_critical_points':     n_crit,
        'mean_vx': mean_vx, 'sd_vx': sd_vx, 'max_vx': max_vx, 'min_vx': min_vx,
        'mean_vy': mean_vy, 'sd_vy': sd_vy, 'max_vy': max_vy, 'min_vy': min_vy,
        'mean_v':  mean_v,  'sd_v':  sd_v,  'max_v':  max_v,  'min_v':  min_v,
        'mean_a':  mean_a,  'sd_a':  sd_a,  'max_a':  max_a,  'min_a':  min_a,
        'mean_jerk': mean_j, 'sd_jerk': sd_j, 'max_jerk': max_j, 'min_jerk': min_j,
        'a_beg_time': a_beg,
    }


# ── Session-level extraction ───────────────────────────────────────────────────

def extract_session_features(filepath, user_id, window_size=None,
                              more_scroll=False, dir_scroll=False):
    """
    Same as extract_features.py, but each row also carries scroll features.
    If more_scroll=True, also computes deltaY-based features.
    If dir_scroll=True, also computes directional (up/down) burst features.
    """
    session_name = os.path.basename(filepath)
    raw_df  = load_session(filepath)
    actions = segment_actions(raw_df)

    rows = []

    if window_size is None:
        sf = extract_scroll_features(raw_df, 0, len(raw_df),
                                     more_scroll=more_scroll, dir_scroll=dir_scroll)
        for action in actions:
            ts   = compute_time_series(action['events'])
            feat = extract_action_features(action['type'], ts)
            if feat is not None:
                feat['userid']  = user_id
                feat['session'] = session_name
                feat.update(sf)
                rows.append(feat)
    else:
        for w_idx, start in enumerate(range(0, len(actions), window_size)):
            window = actions[start:start + window_size]
            if len(window) < 5:
                continue
            win_label = f'{session_name}_w{w_idx}'

            raw_start = window[0].get('raw_start', 0)
            raw_end   = window[-1].get('raw_end', len(raw_df))
            sf = extract_scroll_features(raw_df, raw_start, raw_end,
                                         more_scroll=more_scroll, dir_scroll=dir_scroll)

            for action in window:
                ts   = compute_time_series(action['events'])
                feat = extract_action_features(action['type'], ts)
                if feat is not None:
                    feat['userid']  = user_id
                    feat['session'] = win_label
                    feat.update(sf)
                    rows.append(feat)

    return pd.DataFrame(rows)


def extract_user_features(user_dir, user_id, window_size=None,
                           more_scroll=False, dir_scroll=False):
    all_rows = []
    for fname in sorted(os.listdir(user_dir)):
        fpath = os.path.join(user_dir, fname)
        df = extract_session_features(fpath, user_id, window_size=window_size,
                                      more_scroll=more_scroll, dir_scroll=dir_scroll)
        all_rows.append(df)
    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract 39 mouse dynamics + scroll features from Balabit-format data.')
    parser.add_argument('--mode', choices=['dataset', 'session'], required=True)
    parser.add_argument('--data_dir',    default='training_files')
    parser.add_argument('--file',        default=None)
    parser.add_argument('--user',        default='unknown')
    parser.add_argument('--out',         default='features_scroll.csv')
    parser.add_argument('--users',       nargs='+', default=None)
    parser.add_argument('--window_size', type=int, default=None)
    parser.add_argument('--more_scroll', action='store_true',
                        help='Include deltaY-based scroll features (collected data only)')
    parser.add_argument('--dir_scroll',  action='store_true',
                        help='Include directional (up/down) scroll burst features')
    args = parser.parse_args()

    if args.mode == 'dataset':
        users = args.users or sorted(os.listdir(args.data_dir))
        all_dfs = []
        for user in users:
            user_dir = os.path.join(args.data_dir, user)
            if not os.path.isdir(user_dir):
                continue
            print(f"Processing {user}...")
            df = extract_user_features(user_dir, user,
                                       window_size=args.window_size,
                                       more_scroll=args.more_scroll,
                                       dir_scroll=args.dir_scroll)
            print(f"  -> {len(df)} actions, {df['session'].nunique()} sessions")
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
        df = extract_session_features(args.file, args.user,
                                      more_scroll=args.more_scroll,
                                      dir_scroll=args.dir_scroll)
        df.to_csv(args.out, index=False)
        print(f"Saved {len(df)} actions to {args.out}")


if __name__ == '__main__':
    main()