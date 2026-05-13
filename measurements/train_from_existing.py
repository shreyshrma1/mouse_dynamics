import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.continual_trainer_39 import ContinualTrainer39

user_id = input("Enter user ID: ").strip()

trainer = ContinualTrainer39(user_id=user_id)
trainer.load()
trainer.backfill('collected_data')

print("\n" + trainer.enrollment_status)