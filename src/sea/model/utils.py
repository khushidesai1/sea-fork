"""
Taken from Facebook's DINO
"""

import numpy as np
import torch.nn as nn
import math

import torch


def get_params_groups(model, args):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    groups = [{"params": regularized,
               "weight_decay": args.weight_decay,
               "lr": args.lr},
              {"params": not_regularized,
               "weight_decay": 0.,
               "lr": args.lr},
               ]
    return groups


def shd_metric(adj_pred, adj_true):
    """
    Calculates the structural hamming distance.
    Copied from:
    https://github.com/slachapelle/dcdi/blob/594d328eae7795785e0d1a1138945e28a4fec037/dcdi/utils/metrics.py

    Args:
        adj_pred: has shape N x N, the predicted adjacency matrix
        adj_true: has shape N x N, the true adjacency matrix
    Returns:
        shd: int, the structural hamming distance
    """
    adj_diff = adj_true - adj_pred

    reversed_edges = (((adj_diff + adj_diff.T) == 0) & (adj_diff != 0)).sum() / 2

    # Each reversed edge necessarily leads to one false positive and one false negative so we need to subtract those
    false_negative = (adj_diff == 1).sum() - reversed_edges
    false_positive = (adj_diff == -1).sum() - reversed_edges

    return false_negative + false_positive + reversed_edges


def sid_metric(adj_pred, adj_true):
    """
    Calculates the structural intervention distance.
    Args:
        adj_pred: has shape N x N, the predicted adjacency matrix
        adj_true: has shape N x N, the true adjacency matrix
    Returns:
        sid: float, the structural intervention distance
    """
    pass


def precision(adj_pred, adj_true):
    """
    Calculates the precision of the predicted adjacency matrix.
    Args:
        adj_pred: has shape N x N, the predicted adjacency matrix
        adj_true: has shape N x N, the true adjacency matrix
    Returns:
        precision: float, the precision of the predicted adjacency matrix
    """
    pred = adj_pred != 0
    true = adj_true != 0

    tp = (pred & true).sum()
    fp = (pred & ~true).sum()

    denom = tp + fp
    zero = np.zeros(())
    prec = np.where(denom > 0, tp / denom, zero)
    return prec


def recall(adj_pred, adj_true):
    """
    Calculates the recall of the predicted adjacency matrix.
    Args:
        adj_pred: has shape N x N, the predicted adjacency matrix
        adj_true: has shape N x N, the true adjacency matrix
    Returns:
        recall: float, the recall of the predicted adjacency matrix
    """
    pred = adj_pred != 0
    true = adj_true != 0

    tp = (pred & true).sum()
    fn = ((~pred) & true).sum()

    denom = tp + fn
    zero = np.zeros(())
    rec = np.where(denom > 0, tp / denom, zero)
    return rec


def f1_score(adj_pred, adj_true):
    """
    Calculates the F1 score of the predicted adjacency matrix.
    Args:
        adj_pred: has shape N x N, the predicted adjacency matrix
        adj_true: has shape N x N, the true adjacency matrix
    Returns:
        f1_score: float, the F1 score of the predicted adjacency matrix
    """
    # Compute precision and recall using the same binarization rules
    pred = adj_pred != 0
    true = adj_true != 0

    tp = (pred & true).sum()
    fp = (pred & ~true).sum()
    fn = ((~pred) & true).sum()

    prec_denom = tp + fp
    rec_denom = tp + fn
    zero = np.zeros(1)
    prec = np.where(prec_denom > 0, tp / prec_denom, zero)
    rec = np.where(rec_denom > 0, tp / rec_denom, zero)

    denom = prec + rec
    f1 = np.where(denom > 0, 2.0 * prec * rec / denom, zero)
    return f1

def to_2d(a):
    options = [10, 11, 20, 100, 200, 300, 400, 500, 1000]
    # find real size
    for n in options:
        if len(a) == n*(n-1):
            break
    
    # align to 2d
    mask = np.tri(n, k=-1, dtype=bool)
    halfway = n*(n-1)//2

    g1 = np.zeros((n, n))
    g2 = np.zeros((n, n))
    g1[mask] = a[:halfway]
    g2[mask] = a[halfway:]

    g = g1 + g2.T
    return g, a[:halfway], a[halfway:]

def to_1d(a):
    n = a.shape[0]
    mask = np.tri(n, k=-1, dtype=bool)
    forward = a[mask]
    backward = a.T[mask]
    return forward, backward

def compute_additional_metrics(true, pred, threshold=0.5):
    # convert to 2d and get aligned edges
    true, pred = np.array(true, dtype=int), np.array(pred)
    if true.ndim == 1:
        true, true_f, true_b = to_2d(true)
        pred, pred_f, pred_b = to_2d(pred)
    else:
        true_f, true_b = to_1d(true)
        pred_f, pred_b = to_1d(pred)
    pred_bin = (pred > threshold).astype(int)
    # compute metrics now
    shd = shd_metric(pred_bin, true)
    f1 = f1_score(pred_bin, true)
    prec_score = precision(pred_bin, true)
    recall_score = recall(pred_bin, true)
    true_mask = (true_f + true_b) > 0
    true_direction = (true_f[true_mask] > true_b[true_mask])
    pred_forward = (pred_f[true_mask] > pred_b[true_mask])
    pred_backward = (pred_f[true_mask] < pred_b[true_mask])
        
    return shd, f1, prec_score, recall_score

