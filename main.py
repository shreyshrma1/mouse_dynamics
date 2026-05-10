import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mouse_dynamics.data_collection.collector import MouseCollector


def main():
    user_id = input("Enter your user ID: ").strip()

    if not user_id:
        print("Error: user ID cannot be empty.")
        return

    collector = MouseCollector(
        user_id=user_id,
        save_dir='collected_data',
        flush_interval=300  # flush every 5 minutes
    )

    print(f"\nCollecting mouse data for '{user_id}'.")
    print(f"Data will be saved to: collected_data/{user_id}/")
    print(f"Sessions are flushed to disk every {collector.flush_interval} seconds.")
    print("Press Ctrl+C to stop.\n")

    collector.start(trainer=None)


if __name__ == '__main__':
    main()