import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from measurements.model import DynamicsClassifier, preprocess_data

from util.settings import *
from util.process import *
from util.const import *
from util.utils import datasetname, create_userids, keeporder_split


def build_binary_dataset(dataset, target_user):
    num_features = dataset.shape[1]
    x = dataset.iloc[:, 0: num_features - 1].values
    userids = dataset["userid"].values

    legitimate_mask = userids == target_user
    impostor_mask = ~legitimate_mask

    x_legit = x[legitimate_mask]
    x_impostor = x[impostor_mask]

    n_legit = len(x_legit)
    rng = np.random.RandomState(0)
    impostor_indices = rng.choice(len(x_impostor), size=n_legit, replace=False)
    x_impostor_sampled = x_impostor[impostor_indices]

    x_combined = np.vstack([x_legit, x_impostor_sampled])
    y_combined = np.array([1] * n_legit + [0] * n_legit)

    return x_combined, y_combined


def compute_auc_and_threshold(y_true, y_scores):
    """
    Compute AUC and find the optimal threshold using EER,
    matching the same method used by the Random Forest baseline.
    Returns auc_score, eer_threshold, fpr, tpr.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    # find EER threshold — the point where FAR == FRR
    # brentq finds the root of (1 - tpr) - fpr = 0
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = float(interp1d(fpr, thresholds)(eer))
    except (ValueError, ZeroDivisionError):
        eer_threshold = 0.5  # fallback if EER can't be computed

    return auc_score, eer_threshold, fpr, tpr


def train_model(current_dataset, dataset_amount, num_actions, num_training_actions,
                target_user, lr=1e-5, num_epochs=0, batch_size=64):

    filename = FEAT_DIR + '/' + datasetname(current_dataset, dataset_amount, num_training_actions)
    dataset = pd.read_csv(filename)

    x, y = build_binary_dataset(dataset, target_user)

    if CURRENT_SPLIT_TYPE == SPLIT_TYPE.RANDOM:
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    else:
        x_train, x_val, y_train, y_val = keeporder_split(x, y, test_size=TEST_SIZE)

    scaler, x_train_t, x_val_t = preprocess_data(x_train, x_val)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    model = DynamicsClassifier()

    checkpoint_dir = 'checkpoints'
    checkpoint_path = f'{checkpoint_dir}/dynamics_classifier_{target_user}.pt'
    os.makedirs(checkpoint_dir, exist_ok=True)
    joblib.dump(scaler, f'{checkpoint_dir}/scaler_{target_user}.pkl')

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded existing checkpoint for {target_user}")

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(x_train_t.size(0))
        epoch_loss = 0
        for i in range(0, x_train_t.size(0), batch_size):
            indices = perm[i: i + batch_size]
            batch_x = x_train_t[indices]
            batch_y = y_train_t[indices]
            optimizer.zero_grad()
            loss = loss_func(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = loss_func(val_logits, y_val_t)

            # get probabilities for AUC computation
            y_scores = torch.sigmoid(val_logits).squeeze(1).numpy()
            y_true = y_val_t.squeeze(1).numpy()

            # compute AUC and EER threshold
            auc_score, eer_threshold, _, _ = compute_auc_and_threshold(y_true, y_scores)

            # use EER threshold for accuracy/FAR/FRR — matches Random Forest baseline
            val_preds = (y_scores >= eer_threshold).astype(float)
            val_acc = (val_preds == y_true).mean()

            tp = ((val_preds == 1) & (y_true == 1)).sum()
            fp = ((val_preds == 1) & (y_true == 0)).sum()
            fn = ((val_preds == 0) & (y_true == 1)).sum()
            tn = ((val_preds == 0) & (y_true == 0)).sum()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved new best model for {target_user} (val loss: {val_loss:.4f})")

        print(f"[{target_user}] Epoch {epoch+1}/{num_epochs} "
              f"| Train Loss: {epoch_loss / (x_train_t.size(0) // batch_size):.4f} "
              f"| Val Loss: {val_loss:.4f} "
              f"| Val Acc: {val_acc:.4f} "
              f"| AUC: {auc_score:.4f} "
              f"| EER Threshold: {eer_threshold:.4f} "
              f"| FAR: {far:.4f} | FRR: {frr:.4f}")

    return scaler, model


def train_all_users(current_dataset, dataset_amount, num_actions, num_training_actions):
    filename = FEAT_DIR + '/' + datasetname(current_dataset, dataset_amount, num_training_actions)
    dataset = pd.read_csv(filename)
    userids = list(create_userids(current_dataset))

    results = {}
    for user in userids:
        print(f"\n{'='*50}")
        print(f"Training classifier for user: {user}")
        print(f"{'='*50}")
        scaler, model = train_model(
            current_dataset, dataset_amount, num_actions,
            num_training_actions, target_user=user
        )
        results[user] = (scaler, model)

    return results


def evaluate_model(target_user, current_dataset, dataset_amount, num_training_actions):
    checkpoint_path = f'checkpoints/dynamics_classifier_{target_user}.pt'
    scaler_path = f'checkpoints/scaler_{target_user}.pkl'
    if not os.path.exists(checkpoint_path) or not os.path.exists(scaler_path):
        return None

    filename = FEAT_DIR + '/' + datasetname(current_dataset, dataset_amount, num_training_actions)
    dataset = pd.read_csv(filename)
    x, y = build_binary_dataset(dataset, target_user)

    if CURRENT_SPLIT_TYPE == SPLIT_TYPE.RANDOM:
        _, x_val, _, y_val = train_test_split(x, y, test_size=TEST_SIZE, random_state=0)
    else:
        _, x_val, _, y_val = keeporder_split(x, y, test_size=TEST_SIZE)

    scaler = joblib.load(scaler_path)
    x_val_t = torch.FloatTensor(scaler.transform(x_val))
    y_true = y_val.astype(float)

    model = DynamicsClassifier()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        val_logits = model(x_val_t)
        y_scores = torch.sigmoid(val_logits).squeeze(1).numpy()

    # compute AUC and EER threshold
    auc_score, eer_threshold, fpr, tpr = compute_auc_and_threshold(y_true, y_scores)

    # apply EER threshold for accuracy/FAR/FRR
    preds = (y_scores >= eer_threshold).astype(float)
    acc = (preds == y_true).mean()
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    tn = ((preds == 0) & (y_true == 0)).sum()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        'accuracy': acc,
        'AUC': auc_score,
        'EER_threshold': eer_threshold,
        'FAR': far,
        'FRR': frr,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'fpr': fpr, 'tpr': tpr  # kept for plotting ROC curves later
    }


def evaluate_all_users(current_dataset, dataset_amount, num_training_actions):
    userids = list(create_userids(current_dataset))
    print(f"\n{'User':<10} {'Accuracy':>10} {'AUC':>8} {'EER Thr':>10} {'FAR':>8} {'FRR':>8}")
    print('-' * 58)
    accs, aucs, fars, frrs = [], [], [], []
    for user in userids:
        result = evaluate_model(user, current_dataset, dataset_amount, num_training_actions)
        if result is None:
            print(f"{str(user):<10} {'(no checkpoint)':>46}")
            continue
        accs.append(result['accuracy'])
        aucs.append(result['AUC'])
        fars.append(result['FAR'])
        frrs.append(result['FRR'])
        print(f"{str(user):<10} {result['accuracy']:>10.4f} {result['AUC']:>8.4f} "
              f"{result['EER_threshold']:>10.4f} {result['FAR']:>8.4f} {result['FRR']:>8.4f}")
    if accs:
        print('-' * 58)
        print(f"{'Mean':<10} {np.mean(accs):>10.4f} {np.mean(aucs):>8.4f} "
              f"{'':>10} {np.mean(fars):>8.4f} {np.mean(frrs):>8.4f}")


if __name__ == '__main__':
    train_all_users(
        CURRENT_DATASET,
        DATASET_USAGE,
        NUM_ACTIONS,
        NUM_TRAINING_SAMPLES
    )
    evaluate_all_users(CURRENT_DATASET, DATASET_USAGE, NUM_TRAINING_SAMPLES)