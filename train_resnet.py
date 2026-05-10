import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from measurements.time_resnet import ResNetTime
from measurements.time_series_loader import BalabitDataset

USERS = ['user7', 'user9', 'user12', 'user15', 'user16',
         'user20', 'user21', 'user23', 'user29', 'user35']
TRAINING_DIR = '/scratch/gilbreth/shar1159/cynics/mouse_dynamics/balabit_dataset/training_files'
TEST_DIR = '/scratch/gilbreth/shar1159/cynics/mouse_dynamics/balabit_dataset/test_files'
CHECKPOINT_DIR = 'checkpoints'
BLOCK_SIZE = 128
BATCH_SIZE = 64
LR = 1e-3
NUM_EPOCHS = 100
TEST_SIZE = 0.33

def compute_auc_and_threshold(gt, y):
    fpr, tpr, thresholds = roc_curve(gt, y)
    auc_score = auc(fpr, tpr)
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = float(interp1d(fpr, thresholds)(eer))
    except (ValueError, ZeroDivisionError):
        eer_threshold = float(np.median(thresholds))
    return auc_score, eer_threshold, fpr, tpr

def train_user(user, data_dir=TRAINING_DIR):
    train_dataset = BalabitDataset(user, data_dir=TRAINING_DIR,
                                   block_size=BLOCK_SIZE, mode="legitimate")
    eval_dataset = BalabitDataset(user, data_dir=TRAINING_DIR,
                                  block_size=BLOCK_SIZE, mode="binary")
    val_size = int(TEST_SIZE * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_set, _ = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(0)
    )
    eval_val_size = int(TEST_SIZE * len(eval_dataset))
    eval_train_size = len(eval_dataset) - eval_val_size
    _, eval_val_set = random_split(
        eval_dataset, [eval_train_size, eval_val_size],
        generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_val_set, batch_size=BATCH_SIZE, shuffle=False)
    model = ResNetTime(seq_len=BLOCK_SIZE)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = f'{CHECKPOINT_DIR}/autoencoder_{user}.pt'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded existing checkpoint for {user}")
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_train_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        num_batches = 0
        for x in train_loader:
            optimizer.zero_grad()
            x_recon = model(x)
            loss = loss_func(x_recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        avg_train_loss = epoch_loss / num_batches

        model.eval()
        all_errors = []
        all_labels = []
        with torch.no_grad():
            for x, y in eval_loader:
                errors = model.reconstruction_error(x)
                all_errors.extend(errors.numpy())
                all_labels.extend(y.numpy())
        all_errors = np.array(all_errors)
        all_labels = np.array(all_labels)
        auc_score, eer_threshold, _, _ = compute_auc_and_threshold(
            all_labels, -all_errors
        )
        y_preds = (-all_errors >= eer_threshold).astype(int)
        val_acc = (y_preds == all_labels).mean()
        tp = ((y_preds == 1) & (all_labels == 1)).sum()
        fp = ((y_preds == 1) & (all_labels == 0)).sum()
        fn = ((y_preds == 0) & (all_labels == 1)).sum()
        tn = ((y_preds == 0) & (all_labels == 0)).sum()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved new best model (train loss: {avg_train_loss:.6f})")
        print(f"[{user}] Epoch {epoch+1}/{NUM_EPOCHS} "
              f"| Train Loss: {avg_train_loss:.6f} "
              f"| AUC: {auc_score:.4f} "
              f"| EER Thr: {eer_threshold:.4f} "
              f"| Val Acc: {val_acc:.4f} "
              f"| FAR: {far:.4f} | FRR: {frr:.4f}")
    return model

def evaluate_user(target_user, data_dir=TRAINING_DIR):
    checkpoint_path = f'{CHECKPOINT_DIR}/autoencoder_{target_user}.pt'
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found for {target_user}")
        return None

    eval_dataset = BalabitDataset(target_user, data_dir=data_dir,
                                  block_size=BLOCK_SIZE, mode='binary')
    eval_val_size = int(TEST_SIZE * len(eval_dataset))
    eval_train_size = len(eval_dataset) - eval_val_size
    _, eval_val_set = random_split(
        eval_dataset, [eval_train_size, eval_val_size],
        generator=torch.Generator().manual_seed(0)
    )
    eval_loader = DataLoader(eval_val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = ResNetTime(seq_len=BLOCK_SIZE)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    all_errors = []
    all_labels = []
    with torch.no_grad():
        for x, y in eval_loader:
            errors = model.reconstruction_error(x)
            all_errors.extend(errors.numpy())
            all_labels.extend(y.numpy())

    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)

    auc_score, eer_threshold, fpr, tpr = compute_auc_and_threshold(
        all_labels, -all_errors
    )

    y_preds = (-all_errors >= eer_threshold).astype(int)
    acc = (y_preds == all_labels).mean()
    tp = ((y_preds == 1) & (all_labels == 1)).sum()
    fp = ((y_preds == 1) & (all_labels == 0)).sum()
    fn = ((y_preds == 0) & (all_labels == 1)).sum()
    tn = ((y_preds == 0) & (all_labels == 0)).sum()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        'accuracy': acc,
        'AUC': auc_score,
        'EER_threshold': eer_threshold,
        'FAR': far,
        'FRR': frr,
        'fpr': fpr,
        'tpr': tpr
    }

def train_all_users():
    for user in USERS:
        train_user(user)


def evaluate_all_users():
    print(f"\n{'User':<10} {'Accuracy':>10} {'AUC':>8} {'EER Thr':>10} {'FAR':>8} {'FRR':>8}")
    print('-' * 58)
    accs, aucs, fars, frrs = [], [], [], []
    for user in USERS:
        result = evaluate_user(user)
        if result is None:
            print(f"{user:<10} {'(no checkpoint)':>46}")
            continue
        accs.append(result['accuracy'])
        aucs.append(result['AUC'])
        fars.append(result['FAR'])
        frrs.append(result['FRR'])
        print(f"{user:<10} {result['accuracy']:>10.4f} {result['AUC']:>8.4f} "
              f"{result['EER_threshold']:>10.4f} {result['FAR']:>8.4f} {result['FRR']:>8.4f}")
    if accs:
        print('-' * 58)
        print(f"{'Mean':<10} {np.mean(accs):>10.4f} {np.mean(aucs):>8.4f} "
              f"{'':>10} {np.mean(fars):>8.4f} {np.mean(frrs):>8.4f}")

if __name__ == '__main__':
    train_all_users()
    evaluate_all_users()
