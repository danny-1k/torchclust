import torch
import torch.nn.functional as F


class DBSCAN:
    def __init__(
        self,
        min_samples: int,
        eps: float,
        seed: int = 42,
        distance: str = "euclidean",
        device: str = "cpu",
    ):
        """Implements the DBSCAN algorithm.

        For every point in the dataset, the closest points are calculated (points that are less than eps away).
        If the point has more than `min_samples` neighbours, then it is considered a core point and a new cluster is formed.
        For each of the other points in the newly formed cluster, we determine the number of neighbours they have.
        If the points have fewer than `min_samples` points, then it is considered a "leaf node" otherwise, all its neigbours are added to the new cluster.

        At the end of the clustering, any point that is not assigned to a cluster is given a "-1" label signifying that it is a noise point or rather an outlier.

        Args:
            min_samples (int): Minimum number of samples required to form a cluster.
            eps (float): Maximum distance between a point and another point to be considered a neighbour.
            seed (int, optional): Random Seed for reproducibility. Defaults to 42.
            distance (str, optional): Distance metric. Can be one of ["euclidean", "cosine"]. Defaults to "euclidean".
            device (str, optional): Device to run clustering on. Defaults to "cpu".
        """
        self.min_samples = min_samples
        self.eps = eps
        self.seed = seed
        self.distance = distance
        self.device = device

        assert distance.lower() in [
            "euclidean",
            "cosine",
        ], '`distance` must be one of "euclidean", "cosine"'

        self._distance_metric = (
            torch._euclidean_dist if distance == "euclidean" else F.cosine_similarity
        )
        self._seed()

    def _seed(self):
        torch.manual_seed(self.seed)

    @torch.no_grad()
    def fit_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the cluster labels for each sample .

        Args:
            x (torch.Tensor): Dataset.

        Returns:
            torch.Tensor: Cluster labels.
        """
        x = x.to(self.device).float()
        indices = torch.arange(x.shape[0]).to(self.device)
        labels = torch.zeros((x.shape[0]), dtype=int).fill_(-1).to(self.device)

        current_cluster_label = -1

        for index in range(x.shape[0]):
            if labels[index] != -1:
                continue

            distances = self._distance_metric(x[index].unsqueeze(0), x)[0]

            neighbours = indices[distances < self.eps]
            neighbours_labels = labels[neighbours]
            neigbours_clusters_assigned = neighbours[neighbours_labels != -1]
            neigbours_clusters_unassigned = neighbours[neighbours_labels == -1]

            if neigbours_clusters_assigned.shape[0] > 0:
                neighbours_labels = neighbours_labels[neighbours_labels != -1]
                closest_cluster = distances[neigbours_clusters_assigned].argmin()
                labels[index] = neighbours_labels[closest_cluster]

                labels[neigbours_clusters_unassigned] = labels[index]

            elif neighbours.shape[0] >= self.min_samples:
                current_cluster_label += 1
                labels[index] = current_cluster_label
                labels[neighbours] = current_cluster_label

        return labels
