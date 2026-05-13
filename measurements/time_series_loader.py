import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset

COLUMNS = ['record_timestamp', 'client_timestamp', 'button', 'state', 'x', 'y']
BLOCK_SIZE = 128
TRAINING_DIR = '/scratch/gilbreth/shar1159/cynics/mouse_dynamics/balabit_dataset/training_files'
USERS = ['user7', 'user9', 'user12', 'user15', 'user16',
         'user20', 'user21', 'user23', 'user29', 'user35']


def load_user_data(user, data_dir=TRAINING_DIR):
    user_dir = os.path.join(data_dir, user)
    dataframes = []
    for filename in sorted(os.listdir(user_dir)):
        filepath = os.path.join(user_dir, filename)
        df = pd.read_csv(filepath, names=COLUMNS, header=0)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def compute_velocities(df):
    df = df[df["state"] != "Scroll"].copy()
    df = df[df["state"] != "Down"].copy()
    dt = df["client_timestamp"].diff()
    dx = df["x"].diff()
    dy = df["y"].diff()
    df["vx"] = dx / dt.replace(0, np.nan)
    df["vy"] = dy / dt.replace(0, np.nan)
    df = df.dropna(subset=['vx', 'vy'])
    df = df[np.isfinite(df['vx']) & np.isfinite(df['vy'])]

    # ── normalize velocities ───────────────────────────────────────────
    vx = df['vx'].values
    vy = df['vy'].values

    # clip extreme outliers first (1st and 99th percentile)
    vx = np.clip(vx, np.percentile(vx, 1), np.percentile(vx, 99))
    vy = np.clip(vy, np.percentile(vy, 1), np.percentile(vy, 99))

    # normalize to zero mean unit variance
    vx = (vx - vx.mean()) / (vx.std() + 1e-8)
    vy = (vy - vy.mean()) / (vy.std() + 1e-8)

    df = df.copy()
    df['vx'] = vx
    df['vy'] = vy


    return df[['vx', 'vy']].values


def make_blocks(velocity_array, block_size=BLOCK_SIZE):
    n = len(velocity_array)
    num_blocks = n // block_size
    if num_blocks == 0:
        return None
    trimmed = velocity_array[:num_blocks * block_size]
    return trimmed.reshape(num_blocks, block_size, 2)


class BalabitDataset(Dataset):
    def __init__(self, target_user, data_dir=TRAINING_DIR, block_size=BLOCK_SIZE,
                 mode='binary'):
        self.mode = mode
        real_blocks = []
        fake_blocks = []

        for user in USERS:
            df = load_user_data(user, data_dir)
            velocities = compute_velocities(df)
            blocks = make_blocks(velocities, block_size)
            if blocks is None:
                continue
            if user == target_user:
                real_blocks.append(blocks)
            else:
                fake_blocks.append(blocks)

        real_blocks = np.vstack(real_blocks)

        if mode == 'legitimate':
            self.sequences = torch.FloatTensor(real_blocks)
        else:
            fake_blocks = np.vstack(fake_blocks)
            n_real = len(real_blocks)
            fake_indices = np.random.choice(
                len(fake_blocks), size=n_real, replace=False
            )
            fake_blocks_sampled = fake_blocks[fake_indices]
            all_blocks = np.vstack([real_blocks, fake_blocks_sampled])
            all_labels = np.array([1] * n_real + [0] * n_real)
            perm = np.random.permutation(len(all_blocks))
            self.sequences = torch.FloatTensor(all_blocks[perm])
            self.labels = torch.LongTensor(all_labels[perm])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.mode == 'legitimate':
            return self.sequences[idx]
        return self.sequences[idx], self.labels[idx]