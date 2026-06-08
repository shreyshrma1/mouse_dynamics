import numpy as np
import pandas as pd

COLUMNS = ['record_timestamp', 'client_timestamp', 'button', 'state', 'x', 'y']

BUTTON_CATS = ['Move', 'Left', 'Right', 'Scroll']
STATE_CATS  = ['Move', 'Pressed', 'Released', 'Up', 'Down']

INPUT_DIM = 3 + len(BUTTON_CATS) + len(STATE_CATS)  # 12


def load_session(filepath):
    df = pd.read_csv(filepath, names=COLUMNS, header=None)
    df = df[df['record_timestamp'] != 'record_timestamp']
    df['client_timestamp'] = pd.to_numeric(df['client_timestamp'], errors='coerce')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['client_timestamp', 'x', 'y']).reset_index(drop=True)
    return df


def encode_events(df):
    n = len(df)

    t = df['client_timestamp'].values.astype(np.float32)
    x = df['x'].values.astype(np.float32)
    y = df['y'].values.astype(np.float32)

    dt = np.zeros(n, dtype=np.float32)
    dx = np.zeros(n, dtype=np.float32)
    dy = np.zeros(n, dtype=np.float32)
    dt[1:] = np.diff(t)
    dx[1:] = np.diff(x)
    dy[1:] = np.diff(y)

    button_oh = np.zeros((n, len(BUTTON_CATS)), dtype=np.float32)
    for i, cat in enumerate(BUTTON_CATS):
        button_oh[:, i] = (df['button'].values == cat).astype(np.float32)

    state_oh = np.zeros((n, len(STATE_CATS)), dtype=np.float32)
    for i, cat in enumerate(STATE_CATS):
        state_oh[:, i] = (df['state'].values == cat).astype(np.float32)

    features = np.stack([dt, dx, dy], axis=1)
    features = np.concatenate([features, button_oh, state_oh], axis=1)
    return features


def extract_raw_windows(session_files, window_size=200, stride=None):
    if stride is None:
        stride = window_size // 2

    all_windows = []
    for path in session_files:
        try:
            df = load_session(path)
            if len(df) < window_size:
                continue
            events = encode_events(df)
            for start in range(0, len(events) - window_size + 1, stride):
                window = events[start:start + window_size]
                all_windows.append(window)
        except Exception as e:
            import os
            print(f"  [!] {os.path.basename(path)}: {e}", flush=True)

    return all_windows