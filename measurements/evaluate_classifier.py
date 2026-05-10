import pandas as pd
import warnings
import copy
import sys
import numpy as np

from itertools import cycle
from sklearn import model_selection, metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from util.myplots import plotROCs
from util.settings import *
from util.process import *
from util.const import *

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from util.utils import datasetname, create_userids, keeporder_split


def evaluate_dataset(current_dataset, dataset_amount, num_actions, num_training_actions):
    filename = FEAT_DIR + '/' + datasetname(current_dataset, dataset_amount, num_training_actions)

    print(filename)
    dataset = pd.read_csv(filename)
    print(dataset.shape)

    df = pd.DataFrame(dataset)

    num_features = int(dataset.shape[1])
    print("Num features: ", num_features)

    userids = create_userids(current_dataset)
    print(userids)

    items = userids
    fpr = {}
    tpr = {}
    roc_auc = {}
    results = {}

    for i in userids:
        user_positive_data = df.loc[df.iloc[:, -1].isin([i])]
        numSamples = user_positive_data.shape[0]
        array_positive = copy.deepcopy(user_positive_data.values)
        array_positive[:, -1] = 1

        user_negative_data = select_negatives_from_other_users(dataset, i, numSamples)
        array_negative = copy.deepcopy(user_negative_data.values)
        array_negative[:, -1] = 0

        dataset_user = pd.concat([pd.DataFrame(array_positive), pd.DataFrame(array_negative)]).values
        X = dataset_user[:, 0:-1]
        y = dataset_user[:, -1]

        if CURRENT_SPLIT_TYPE == SPLIT_TYPE.RANDOM:
            X_train, X_validation, y_train, y_validation = model_selection.train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        else:
            X_train, X_validation, y_train, y_validation = keeporder_split(X, y, test_size=TEST_SIZE)

        model = RandomForestClassifier(random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

        scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=False)
        cv_accuracy = scores['test_score']
        print("CV Accuracy: %0.2f (+/- %0.2f)" % (cv_accuracy.mean(), cv_accuracy.std() * 2))

        y_predicted = model.predict(X_validation)
        test_accuracy = accuracy_score(y_validation, y_predicted)
        print("Test Accuracy: %0.2f" % test_accuracy)

        fpr[i], tpr[i], thr = evaluate_sequence_of_samples(model, X_validation, y_validation, num_actions)

        threshold = -1
        far = None
        frr = None
        try:
            eer = brentq(lambda x: 1. - x - interp1d(fpr[i], tpr[i])(x), 0., 1.)
            threshold = float(interp1d(fpr[i], thr)(eer))

            y_scores_dict = get_scores(model, X_validation, y_validation, num_actions)
            y_preds = (y_scores_dict['scores'] >= threshold).astype(int)
            y_true = y_scores_dict['labels']

            tp = ((y_preds == 1) & (y_true == 1)).sum()
            fp = ((y_preds == 1) & (y_true == 0)).sum()
            fn = ((y_preds == 0) & (y_true == 1)).sum()
            tn = ((y_preds == 0) & (y_true == 0)).sum()

            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        except (ZeroDivisionError, ValueError):
            print("Division by zero")

        roc_auc[i] = auc(fpr[i], tpr[i])

        print(str(i) + ": " + str(roc_auc[i]) + " threshold: " + str(threshold))
        print(f"  FAR: {far:.4f} | FRR: {frr:.4f}")

        results[i] = {
            'auc': roc_auc[i],
            'threshold': threshold,
            'test_accuracy': test_accuracy,
            'far': far,
            'frr': frr,
            'num_samples': numSamples
        }

    # summary table
    print(f"\n{'User':<10} {'Samples':>8} {'AUC':>8} {'EER Thr':>10} {'Test Acc':>10} {'FAR':>8} {'FRR':>8}")
    print('-' * 68)
    aucs, accs, fars, frrs, samples = [], [], [], [], []
    for i in userids:
        r = results[i]
        aucs.append(r['auc'])
        accs.append(r['test_accuracy'])
        samples.append(r['num_samples'])
        if r['far'] is not None:
            fars.append(r['far'])
            frrs.append(r['frr'])
        print(f"{str(i):<10} {r['num_samples']:>8} {r['auc']:>8.4f} {r['threshold']:>10.4f} "
              f"{r['test_accuracy']:>10.4f} {r['far']:>8.4f} {r['frr']:>8.4f}")
    print('-' * 68)
    print(f"{'Mean':<10} {int(sum(samples)/len(samples)):>8} {sum(aucs)/len(aucs):>8.4f} {'':>10} "
          f"{sum(accs)/len(accs):>10.4f} {sum(fars)/len(fars):>8.4f} {sum(frrs)/len(frrs):>8.4f}")

    plotROCs(fpr, tpr, roc_auc, items)
    return results


def get_scores(model, X_validation, y_validation, num_actions):
    """
    Returns raw scores and labels as arrays rather than an ROC curve,
    needed to apply a specific threshold and compute FAR/FRR directly.
    """
    if num_actions == 1:
        y_scores = model.predict_proba(X_validation)[:, 1]
        return {'scores': y_scores, 'labels': y_validation}

    X_val_positive = []
    X_val_negative = []
    for i in range(len(y_validation)):
        if y_validation[i] == 1:
            X_val_positive.append(X_validation[i])
        else:
            X_val_negative.append(X_validation[i])

    pos_scores = model.predict_proba(X_val_positive)
    neg_scores = model.predict_proba(X_val_negative)

    scores = []
    labels = []

    n_pos = len(X_val_positive)
    for i in range(n_pos - num_actions + 1):
        score = sum(pos_scores[i + j][1] for j in range(num_actions)) / num_actions
        scores.append(score)
        labels.append(1)

    n_neg = len(X_val_negative)
    for i in range(n_neg - num_actions + 1):
        score = sum(neg_scores[i + j][1] for j in range(num_actions)) / num_actions
        scores.append(score)
        labels.append(0)

    return {'scores': np.array(scores), 'labels': np.array(labels)}


def evaluate_sequence_of_samples(model, X_validation, y_validation, num_actions):
    if num_actions == 1:
        y_scores = model.predict_proba(X_validation)
        writeCSVa(y_validation, y_scores[:, 1])
        return roc_curve(y_validation, y_scores[:, 1])

    result = get_scores(model, X_validation, y_validation, num_actions)
    return roc_curve(result['labels'], result['scores'])


def select_negatives_from_other_users(dataset, userid, numsamples):
    other_users_data = dataset['userid'] != userid
    dataset_negatives = dataset[other_users_data].sample(numsamples, random_state=RANDOM_STATE)
    return dataset_negatives