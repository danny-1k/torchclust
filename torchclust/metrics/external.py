"""External Clustering Metrics

Metrics that utilise "external" information about the clusters such as the true labels to determine the quality of clustering.

"""

import torch
from torchclust.utils import confusion_matrix


def rand_index(labels_true: torch.Tensor, labels_pred: torch.Tensor):
    """Rand Index Metric.

    Args:
        labels_true (torch.Tensor): Ground Truth class labels.
        labels_pred (torch.Tensor): Predicted class labels from algorithm.
    """

    if not isinstance(labels_true, torch.Tensor) or not isinstance(
        labels_pred, torch.Tensor
    ):
        raise ValueError("Expected pytorch Tensors as parameters.")

    if not len(labels_true.shape) == 1 or not len(labels_pred.shape) == 1:
        raise ValueError("Expected parameters to be of shape (N,)")

    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError(
            f"Expected both labels to be of same shape. Got {labels_true.shape[0]} and {labels_pred.shape[0]} Instead."
        )

    labels_true = labels_true.long()
    labels_pred = labels_pred.long()

    N = labels_true.shape[0]

    a = 0
    b = 0

    for i in range(N):
        for j in range(i + 1, N):
            if (labels_true[i] == labels_true[j]) and (
                labels_pred[i] == labels_pred[j]
            ):
                a += 1

            if (labels_true[i] != labels_true[j]) and (
                labels_pred[i] != labels_pred[j]
            ):
                b += 1

    Nc2 = N * (N - 1) * 0.5

    return (a + b) / Nc2


def purity_score(labels_true: torch.Tensor, labels_pred: torch.Tensor):
    """Purity Score Metric

    Args:
        labels_true (torch.Tensor): Ground Truth class labels.
        labels_pred (torch.Tensor): Predicted class labels from algorithm.
    """

    if not isinstance(labels_true, torch.Tensor) or not isinstance(
        labels_pred, torch.Tensor
    ):
        raise ValueError("Expected pytorch Tensors as parameters.")

    if not len(labels_true.shape) == 1 or not len(labels_pred.shape) == 1:
        raise ValueError("Expected parameters to be of shape (N,)")

    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError(
            f"Expected both labels to be of same shape. Got {labels_true.shape[0]} and {labels_pred.shape[0]} Instead."
        )

    labels_true = labels_true.long()
    labels_pred = labels_pred.long()

    N = labels_true.shape[0]
    k = labels_true.max() + 1

    purity = confusion_matrix(labels_true, labels_pred, k).max(axis=0).values.sum() / N

    return purity.item()


def mutual_information(labels_true: torch.Tensor, labels_pred: torch.Tensor):
    """Mutual Information Metric

    Args:
        labels_true (torch.Tensor): Ground Truth class labels.
        labels_pred (torch.Tensor): Predicted class labels from algorithm.
    """

    if not isinstance(labels_true, torch.Tensor) or not isinstance(
        labels_pred, torch.Tensor
    ):
        raise ValueError("Expected pytorch Tensors as parameters.")

    if not len(labels_true.shape) == 1 or not len(labels_pred.shape) == 1:
        raise ValueError("Expected parameters to be of shape (N,)")

    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError(
            f"Expected both labels to be of same shape. Got {labels_true.shape[0]} and {labels_pred.shape[0]} Instead."
        )

    labels_true = labels_true.long()
    labels_pred = labels_pred.long()

    N = labels_true.shape[0]
    R = labels_true.max() + 1
    C = labels_pred.max() + 1

    MI = 0

    for i in range(R):
        nU = labels_true == i
        Pu = nU.sum() / N

        for j in range(C):
            nV = labels_pred == j
            Pv = nV.sum() / N
            Puv = (nU * nV).sum() / N

            MI += Puv * (Puv.log() - Pu.log() - Pv.log()) if Puv > 0 else 0

    return MI.item()
