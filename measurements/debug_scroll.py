"""
debug_scroll.py

Checks whether scroll features are actually populated in the output of
extract_features_scroll. Run this from your project root.

Usage:
    python debug_scroll.py
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features_scroll import (
    load_session, SCROLL_COLS
)

DATA_DIR = "balabit_dataset/training_files"
BALABIT_USERS = [
    "user7", "user9", "user12", "user15", "user16",
    "user20", "user21", "user23", "user29", "user35",
]

print("=" * 60)
print("1. RAW SCROLL EVENT CHECK")
print("   Checking what 'state' and 'button' values scroll events have")
print("=" * 60)

for user in BALABIT_USERS[:3]:  # just first 3 users
    user_dir = os.path.join(DATA_DIR, user)
    first_file = sorted(os.listdir(user_dir))[0]
    fpath = os.path.join(user_dir, first_file)
    raw = load_session(fpath)

    scroll_rows = raw[raw['state'] == 'Scroll']
    print(f"\n{user} / {first_file}:")
    print(f"  Total raw events: {len(raw)}")
    print(f"  Scroll events (state=='Scroll'): {len(scroll_rows)}")
    print(f"  All unique 'state' values: {raw['state'].unique().tolist()}")
    if len(scroll_rows) > 0:
        print(f"  Scroll 'button' values: {scroll_rows['button'].unique().tolist()}")
        print(f"  Scroll x values (should be 0): {scroll_rows['x'].unique().tolist()[:5]}")
        print(f"  Scroll y values (should be 0): {scroll_rows['y'].unique().tolist()[:5]}")
    else:
        print("  !! No scroll events found with state=='Scroll'")
        print(f"  Raw state value counts:\n{raw['state'].value_counts().to_string()}")

print("\n" + "=" * 60)
print("2. SCROLL FEATURE VALUE CHECK")
print("   Checking if scroll columns are non-zero after extraction")
print("=" * 60)

from extract_features_scroll import extract_session_features

for user in BALABIT_USERS[:3]:
    user_dir  = os.path.join(DATA_DIR, user)
    first_file = sorted(os.listdir(user_dir))[0]
    fpath = os.path.join(user_dir, first_file)

    df = extract_session_features(fpath, user, window_size=50)
    print(f"\n{user} / {first_file}:")
    print(f"  Rows extracted: {len(df)}")
    if len(df) > 0 and all(c in df.columns for c in SCROLL_COLS):
        for col in SCROLL_COLS:
            vals = df[col]
            print(f"  {col:30s}  mean={vals.mean():.4f}  max={vals.max():.4f}  nonzero={int((vals != 0).sum())}/{len(vals)}")
    else:
        missing = [c for c in SCROLL_COLS if c not in df.columns]
        print(f"  !! Missing scroll columns: {missing}")
        print(f"  Columns present: {df.columns.tolist()}")