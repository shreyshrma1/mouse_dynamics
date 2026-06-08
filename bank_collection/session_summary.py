import os
import csv

BANK_DATA_DIR = 'bank-data'

def summarize(user_id):
    folder = os.path.join(BANK_DATA_DIR, user_id)
    if not os.path.exists(folder):
        print(f"No data folder found for '{user_id}' at {folder}")
        return

    session_files = sorted([
        f for f in os.listdir(folder)
        if f.startswith('session_') and f.endswith('.csv')
    ])

    if not session_files:
        print(f"No session files found for '{user_id}'.")
        return

    print(f"\n=== Session Summary for {user_id} ===\n")

    total_minutes = 0.0

    for i, filename in enumerate(session_files, 1):
        path = os.path.join(folder, filename)
        timestamps = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    timestamps.append(float(row[0]))
                except (ValueError, IndexError):
                    continue

        if len(timestamps) < 2:
            print(f"  Session {i} ({filename}): not enough data to calculate duration")
            continue

        duration_minutes = (max(timestamps) - min(timestamps)) / 60
        total_minutes += duration_minutes
        print(f"  Session {i} ({filename}): {duration_minutes:.2f} minutes")

    print(f"\n  Total sessions : {len(session_files)}")
    print(f"  Total time     : {total_minutes:.2f} minutes\n")

if __name__ == '__main__':
    user_id = input("Enter user ID (e.g. user1): ").strip()
    summarize(user_id)