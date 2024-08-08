import torch
from torch.nn import functional as F


class KMeansBase:
    def __init__(
        self,
        num_clusters,
        seed=42,
        distance="euclidean",
        device="cpu",
        max_iter=300,
        tol=1e-4,
    ):
        self.num_clusters = num_clusters
        self.seed = seed
        self.distance = distance
        self.device = device
        self.max_iter = max_iter
        self.tol = tol

        assert distance.lower() in [
            "euclidean",
            "cosine",
        ], '`distance` must be one of "euclidean", "cosine"'

        self._distance_metric = (
            torch._euclidean_dist
            if distance == "euclidean"
            else lambda x1, x2: torch.vmap(lambda x: 1 - F.cosine_similarity(x1, x))(
                x2
            ).T
        )
        self._seed()

        self.parameters = {"centroids": None}

    def _seed(self):
        torch.manual_seed(self.seed)

    def _initialise_centroids(self, x):
        num_samples = x.shape[0]
        indices = torch.randperm(num_samples)[: self.num_clusters]
        centroids = x[indices]

        return centroids

    def _update_centroids(self, centroids, x, closest_clusters):
        raise NotImplementedError()

    @torch.no_grad()
    def fit(self, x):
        x = x.float().to(self.device)

        centroids = self._initialise_centroids(x)
        centroids_old = centroids.clone()

        for _ in range(self.max_iter):
            distances = self._distance_metric(x, centroids)
            closest_clusters = distances.argmin(-1)

            centroids = self._update_centroids(centroids, x, closest_clusters)

            if torch.sqrt((centroids_old - centroids) ** 2).mean() < self.tol:
                break

            centroids_old = centroids.clone()

        self.parameters["centroids"] = centroids

    @torch.no_grad()
    def predict(self, x):
        x = x.float().to(self.device)

        if not isinstance(self.parameters.get("centroids"), type(None)):
            distances = self._distance_metric(x, self.parameters.get("centroids"))
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


class KMeans(KMeansBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_centroids(self, centroids, x, closest_clusters):
        centroids = centroids.clone()

        for k in range(self.num_clusters):
            centroids[k] = x[closest_clusters == k].mean(axis=0)

        return centroids


class KMedians(KMeansBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_centroids(self, centroids, x, closest_clusters):
        centroids = centroids.clone()

        for k in range(self.num_clusters):
            median = x[closest_clusters == k].median(axis=0)
            centroids[k] = median.values

        return centroids
