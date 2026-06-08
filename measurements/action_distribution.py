import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.extract_features_scroll import load_session, segment_actions

ACTION_NAMES = {1: 'Mouse Move', 3: 'Drag & Drop', 4: 'Point & Click'}
BURST_THRESHOLD_S = 0.5  # matches extract_features_scroll.py

DATA_DIR = 'bank_collection/bank-data'

def count_scroll_bursts(raw_df):
    """
    Count directional scroll bursts from raw events.
    A new burst starts when:
      - gap to previous scroll > BURST_THRESHOLD_S, OR
      - direction changes (Up vs Down)
    """
    scrolls = raw_df[raw_df['button'] == 'Scroll'].copy()
    if len(scrolls) == 0:
        return 0, 0

    ts        = scrolls['client_timestamp'].values
    dirs      = scrolls['state'].values  # 'Up' or 'Down'

    up_bursts   = 0
    down_bursts = 0

    # Start first burst
    current_dir = dirs[0]
    if current_dir == 'Up':
        up_bursts += 1
    else:
        down_bursts += 1

    for i in range(1, len(ts)):
        gap_break = (ts[i] - ts[i-1]) > BURST_THRESHOLD_S
        dir_break = dirs[i] != current_dir
        if gap_break or dir_break:
            current_dir = dirs[i]
            if current_dir == 'Up':
                up_bursts += 1
            else:
                down_bursts += 1

    return up_bursts, down_bursts


user_id = input("Enter user ID (or 'all' for all users): ").strip()
users = sorted(os.listdir(DATA_DIR)) if user_id == 'all' else [user_id]

totals     = {1: 0, 3: 0, 4: 0}
scroll_up  = 0
scroll_dn  = 0

for uid in users:
    folder = os.path.join(DATA_DIR, uid)
    if not os.path.isdir(folder):
        print(f"No folder found for {uid}")
        continue
    for fname in sorted(os.listdir(folder)):
        fpath  = os.path.join(folder, fname)
        raw_df = load_session(fpath)

        actions = segment_actions(raw_df)
        for a in actions:
            totals[a['type']] = totals.get(a['type'], 0) + 1

        up, dn  = count_scroll_bursts(raw_df)
        scroll_up += up
        scroll_dn += dn

total_actions = sum(totals.values()) + scroll_up + scroll_dn
print(f"\n=== Action Distribution ({'all users' if user_id == 'all' else user_id}) ===\n")
for action_type, count in totals.items():
    pct = 100 * count / total_actions if total_actions > 0 else 0
    print(f"  {ACTION_NAMES[action_type]:<20} {count:>6}  ({pct:.1f}%)")

for label, count in [('Scroll Up', scroll_up), ('Scroll Down', scroll_dn)]:
    pct = 100 * count / total_actions if total_actions > 0 else 0
    print(f"  {label:<20} {count:>6}  ({pct:.1f}%)")

print(f"\n  {'Total':<20} {total_actions:>6}")