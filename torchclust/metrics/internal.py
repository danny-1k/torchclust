"""Internal Clustering Metrics

Metrics that make utilise "internal" information about the clusters to determine the quality of clustering.
"""

import torch


def silhouette_score(x: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor):
    """Implements Simplified Silhouette.

    Args:
        x (torch.Tensor): Data points.
        labels (torch.Tensor): Cluster label for each point.
        centroids (torch.Tensor): Clustering centroids.
    """

    if not isinstance(x, torch.Tensor) or not isinstance(centroids, torch.Tensor):
        raise ValueError("Expected pytorch Tensors as parameters.")

    if not len(x.shape) == 2 or not len(centroids.shape) == 2:
        raise ValueError(
            "Expected `X`` to be of shape (N, features) and `centroids` to be of shape (clusters, features)"
        )

    points_centroids_distances = torch.vmap(
        lambda point: torch._euclidean_dist(point, centroids)
    )(x)

    a = torch.sum((x - centroids[labels]) ** 2, axis=1).sqrt()

    b = points_centroids_distances.topk(k=2, axis=1, largest=False).values[:, 1]

    s_score = (b - a) / torch.max(a, b)
    s_score = s_score.mean()

    return s_score.item()


def inertia(x: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor):
    """Implements Interia / WCSS Metric (Within-Cluster Sum of Squares).

    Args:
        x (torch.Tensor): Data points
        labels (torch.Tensor): Cluster label for each point.
        centroids (torch.Tensor): Clustering centroids.
    """

    if (
        not isinstance(x, torch.Tensor)
        or not isinstance(centroids, torch.Tensor)
        or not isinstance(labels, torch.Tensor)
    ):
        raise ValueError("Expected pytorch Tensors as parameters.")

    if (
        not len(x.shape) == 2
        or not len(centroids.shape) == 2
        or not labels.shape[0] == x.shape[0]
    ):
        raise ValueError(
            "Expected `X`` to be of shape (N, features), `centroids` to be of shape (clusters, features) and `labels` to be of shape (N)."
        )

    labels = labels.long()

    wcss = torch.sum((x - centroids[labels]) ** 2)

    return wcss.item()


def davies_bouldin_index(
    x: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor
):
    """Implements the Davies-Bouldin Index with p=2 and q=1 (Euclidean case).

    Args:
        x (torch.Tensor): Data points.
        labels (torch.Tensor): Cluster label for each point.
        centroids (torch.Tensor): Clustering centroids.
    """

    if (
        not isinstance(x, torch.Tensor)
        or not isinstance(centroids, torch.Tensor)
        or not isinstance(labels, torch.Tensor)
    ):
        raise ValueError("Expected pytorch Tensors as parameters.")

    if (
        not len(x.shape) == 2
        or not len(centroids.shape) == 2
        or not labels.shape[0] == x.shape[0]
    ):
        raise ValueError(
            "Expected `X`` to be of shape (N, features), `centroids` to be of shape (clusters, features) and `labels` to be of shape (N)."
        )

    labels = labels.long()

    S = torch.zeros((centroids.shape[0])).to(x.device)

    for i in range(centroids.shape[0]):
        S[i] = (
            torch._euclidean_dist(x[labels == i], centroids[i].unsqueeze(0)).sum()
            / (labels == i).sum()
        )

    M = torch.vmap(lambda centroid: torch._euclidean_dist(centroid, centroids))(
        centroids
    )

    R = torch.vmap(lambda point: point + S)(S) / M
    R.fill_diagonal_(-torch.inf)
    D = R.max(axis=1).values

    dbi = D.mean().item()

    return dbi


def between_cluster_sum_of_squares(
    x: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor
):
    """Computes the Weighted sum of squared Euclidean distances between each cluster centroid and the mean of the dataset.

    Args:
        x (torch.Tensor): Data points.
        labels (torch.Tensor): Cluster label for each point.
        centroids (torch.Tensor): Clustering centroids.
    """
    labels = labels.long()

    k = labels.max() + 1
    N = (labels == torch.arange(k).unsqueeze(1)).sum(axis=1)
    C = x.mean(axis=0)

    bcss = torch.sum(torch.sum((centroids - C) ** 2, axis=1) * N)

    return bcss.item()


def calinski_harabasz_index(
    x: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor
):

    labels = labels.long()

    k = labels.max() + 1
    n = x.shape[0]

    wcss = inertia(x, labels, centroids)
    bcss = between_cluster_sum_of_squares(x, labels, centroids)

    chi = (bcss / (k - 1)) / (wcss / (n - k))

    return chi.item()
