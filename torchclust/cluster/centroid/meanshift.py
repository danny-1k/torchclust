import torch
from torch.nn import functional as F


class MeanShift:
    def __init__(
        self,
        band_width: int,
        device: str = "cpu",
        max_iter: int = 400,
        tol: float = 1e-4,
    ):
        """Implements the MeanShift Algorithm.

        Identifies clusters in the dataset by iteratively shifting data points towards the mode (the highest density of data points) of a region.

        Args:
            band_width (int): Standard deviation term in gaussian kernel. Smaller values of band_width leads to less points being considered in a neigbourhood.
            device (str, optional): Device to run clustering on. Defaults to "cpu".
            max_iter (int, optional): Maximum number of iterations. Defaults to 400.
            tol (float, optional): Mininum difference between centroid updates. Defaults to 1e-4.
        """
        self.band_width = band_width
        self.device = device
        self.max_iter = max_iter
        self.tol = tol

        self.parameters = {"centroids": None}

    def _initialise_centroids(self, x):
        centroids = x.clone()
        return centroids

    def _gaussian_kernel(self, distance):
        return torch.exp(-0.5 * (distance / self.band_width) ** 2).div(
            self.band_width * (2 * torch.pi) ** 0.5
        )

    @torch.no_grad()
    def fit(self, x):
        x = x.float().to(self.device)

        centroids = self._initialise_centroids(x)

        for _ in range(self.max_iter):
            pairwise_distances = torch._euclidean_dist(x, centroids)
            weights = self._gaussian_kernel(pairwise_distances)
            new_centroids = torch.sum(
                torch.vmap(lambda weight: x * weight.unsqueeze(1))(weights), axis=1
            ) / weights.sum(axis=1).unsqueeze(1)

            if torch.sqrt((centroids - new_centroids) ** 2).mean() < self.tol:
                break

            centroids = new_centroids

        self.parameters["centroids"] = torch.unique(
            torch.round(centroids, decimals=3), dim=0
        )

    @torch.no_grad()
    def predict(self, x):
        x = x.float().to(self.device)

        if not isinstance(self.parameters.get("centroids"), type(None)):
            distances = torch._euclidean_dist(x, self.parameters.get("centroids"))
            labels = distances.argmin(-1)

            return labels

        raise RuntimeError(
            "`.fit()` must be called before calling `.predict()`. You could try `.fit_predict()`"
        )

    @torch.no_grad()
    def fit_predict(self, x):
        self.fit(x)
        labels = self.predict(x)
        return labels
