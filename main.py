"""
main.py

Entry point for mouse dynamics data collection and continual training.
Starts the MouseCollector and ContinualTrainer together so the model
is updated automatically after each 5-minute session flush.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.collector import MouseCollector
from data_collection.continual_trainer import ContinualTrainer


def main():
    user_id = input("Enter your user ID: ").strip()
    if not user_id:
        print("Error: user ID cannot be empty.")
        return

    # load trainer — resumes from disk if previous sessions exist
    trainer = ContinualTrainer(
        user_id=user_id,
        save_dir="checkpoints_ocsvm",
        buffer_size=50,
        window_size=200,
        nu=0.02,
        gamma=0.01,
    )
    trainer.load()

    print(f"\n--- Enrollment Status for '{user_id}' ---")
    print(trainer.enrollment_status)
    print("----------------------------------------\n")

    collector = MouseCollector(
        user_id=user_id,
        save_dir="collected_data",
        flush_interval=300,
    )

    print(f"Collecting mouse data for '{user_id}'.")
    print(f"Data will be saved to: collected_data/{user_id}/")
    print(f"Model will update every {collector.flush_interval // 60} minutes.")
    if trainer.is_ready:
        print("Model is trained and will score each new session.")
    else:
        remaining = max(
            3 - len(trainer.buffer_39),
            3 - len(trainer.buffer_zheng),
            0
        )
        print(f"Need ~{remaining} more session window(s) before model is ready.")
    print("Press Ctrl+C to stop.\n")

    collector.start(trainer=trainer)


if __name__ == "__main__":
    main()