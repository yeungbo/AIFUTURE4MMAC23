""" 
Copyright 2023 AIFUTURE LLC.
Code for MMAC23 challenge from AIFUTURE Lab.
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score
from sklearn.metrics import r2_score, mean_absolute_error


def classification_metrics(y_true, y_pred):
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    f1 = f1_score(y_true, y_pred, average='macro')
    spe = np.mean(specificity(y_true, y_pred))
    return dict(qwk=qwk, f1=f1, spe=spe)


def specificity(y_true: np.array, y_pred: np.array, classes: set = None):
    if classes is None:
        classes = set(np.concatenate((np.unique(y_true), np.unique(y_pred))))
    specs = []
    for cls in classes:
        y_true_cls = np.array((y_true == cls), np.int)
        y_pred_cls = np.array((y_pred == cls), np.int)
        specs.append(recall_score(y_true_cls, y_pred_cls, pos_label=0))
    return specs


def segmentation_metrics(gt, pred, classId=1):
    gt, pred = gt.flatten(), pred.flatten()
    intersection = np.logical_and(gt == classId, pred == classId)
    dice = (2. * intersection.sum()) / (gt.sum() + pred.sum())
    recall = recall_score(gt, pred, labels=[1], zero_division=0)
    precision = precision_score(gt, pred, labels=[1], zero_division=0)
    return dict(dice=dice, recall=recall, precision=precision)


def regression_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return dict(r2=r2, mae=mae)
