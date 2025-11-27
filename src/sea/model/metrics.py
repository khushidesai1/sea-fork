"""
Metrics for the SEA model.
"""
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

    reversed_edges = (((adj_diff + adj_diff.t()) == 0) & (adj_diff != 0)).sum() / 2

    # Each reversed edge necessarily leads to one false positive and one false negative so we need to subtract those
    false_negative = (adj_diff == 1).sum() - reversed_edges
    false_positive = (adj_diff == -1).sum() - reversed_edges

    return false_negative + false_positive + reversed_edges