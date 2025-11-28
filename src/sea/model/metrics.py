"""
Metrics for the SEA model.
"""

import numpy as np

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
    return float(prec)


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
    return float(rec)


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
    zero = torch.zeros(())
    prec = np.where(prec_denom > 0, tp / prec_denom, zero)
    rec = np.where(rec_denom > 0, tp / rec_denom, zero)

    denom = prec + rec
    f1 = np.where(denom > 0, 2.0 * prec * rec / denom, zero)
    return float(f1)

def shd_metric(adj_pred, adj_true):
    """
    Calculates the structural hamming distance on adjacency matrices.
    Copied from:
    https://github.com/slachapelle/dcdi/blob/594d328eae7795785e0d1a1138945e28a4fec037/dcdi/utils/metrics.py

    Args:
        adj_pred: array-like of shape (N, N), the predicted adjacency matrix
        adj_true: array-like of shape (N, N), the true adjacency matrix

    Returns
    -------
    shd : int
        The structural hamming distance.
    """
    # Work in numpy for convenience
    adj_pred = np.asarray(adj_pred)
    adj_true = np.asarray(adj_true)

    adj_diff = adj_true - adj_pred

    reversed_edges = (((adj_diff + adj_diff.T) == 0) & (adj_diff != 0)).sum() / 2

    # Each reversed edge necessarily leads to one false positive and one false negative
    false_negative = (adj_diff == 1).sum() - reversed_edges
    false_positive = (adj_diff == -1).sum() - reversed_edges

    return false_negative + false_positive + reversed_edges


def _flat_to_adj(a):
    """
    Convert a flat vector of edge scores for both directions into an
    (N, N) adjacency-like matrix, following the convention used in
    ``examples/SEA-results.ipynb``.

    The flat vector is assumed to have length N * (N - 1), corresponding
    to two directed edges per unordered pair (i, j), i != j.
    """
    a = np.asarray(a)
    if a.ndim == 2:
        # Already an adjacency matrix
        return a

    L = a.shape[0]
    # Solve n * (n - 1) = L for n
    n_float = (1.0 + np.sqrt(1.0 + 4.0 * L)) / 2.0
    n = int(round(n_float))
    if n * (n - 1) != L:
        raise ValueError(
            f"Cannot infer number of nodes from flat vector of length {L}."
        )

    mask = np.tri(n, k=-1, dtype=bool)
    half = n * (n - 1) // 2

    g1 = np.zeros((n, n), dtype=a.dtype)
    g2 = np.zeros((n, n), dtype=a.dtype)
    g1[mask] = a[:half]
    g2[mask] = a[half:]

    # Combine into full adjacency matrix; see ``to_2d`` in SEA-results.ipynb
    g = g1 + g2.T
    return g


def shd_from_flat(true, pred, threshold=0.5):
    """
    Compute SHD from flat per-edge scores, mirroring the logic in
    ``compute_additional_metrics`` from ``examples/SEA-results.ipynb``.

    Parameters
    ----------
    true : array-like
        Ground-truth edge indicators, either of shape (N, N) or flat of
        length N * (N - 1).
    pred : array-like
        Predicted edge scores with the same shape convention as ``true``.
    threshold : float, optional
        Threshold applied to ``pred`` after converting to an adjacency
        matrix; defaults to 0.5 as in the original notebook.

    Returns
    -------
    shd : float
        Structural Hamming Distance between the thresholded prediction
        and the ground-truth adjacency.
    """
    true = np.asarray(true)
    pred = np.asarray(pred)

    if true.ndim == 1:
        true_adj = _flat_to_adj(true)
        pred_adj = _flat_to_adj(pred)
    else:
        true_adj = true
        pred_adj = pred

    pred_bin = (pred_adj > threshold).astype(int)
    return shd_metric(pred_bin, true_adj)
