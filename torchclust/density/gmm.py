import torch
from torchclust.centroid import KMeans


class GaussianMixtureModel:
    def __init__(
        self,
        num_clusters: int,
        initialise: str = "random",
        seed: int = 42,
        device: str = "cpu",
        max_iter: int = 300,
        tol: float = 1e-4,
    ):
        """Implements Gaussian Mixture Model

        Iteratively makes estimates for the mean and covariance matrix of a mixture of `num_clusters` multivariate gaussian distributions.

        Args:
            num_clusters (int): Number of gaussians to fit to data.
            initialise (str): How to initialise the initial mean points. Can be one of ["random", "kmeans"]. Defaults to "random".
            seed (int, optional): Random Seed for reproducibility. Defaults to 42.
            device (str, optional): Device to run clustering on. Defaults to "cpu".
            max_iter (int, optional): Maximum number of iterations. Defaults to 300.
            tol (float, optional): Mininum difference between centroid updates. Defaults to 1e-4.
        """
        self.num_clusters = num_clusters
        self.initialise = initialise
        self.seed = seed
        self.device = device
        self.max_iter = max_iter
        self.tol = tol

        assert initialise.lower() in ["random", "kmeans"]

        self.parameters = {"pi": None, "mu": None, "sigma": None}

    def _seed(self):
        torch.manual_seed(self.seed)

    def _initialise_parameters(self, x):
        num_samples = x.shape[0]
        num_features = x.shape[1]

        if self.initialise == "random":
            indices = torch.randperm(num_samples)[: self.num_clusters]
            self.parameters["mu"] = x[indices].to(self.device)
        else:
            kmeans = KMeans(num_clusters=self.num_clusters, device=self.device)
            kmeans.fit(x)
            self.parameters["mu"] = kmeans.parameters["centroids"]

        self.parameters["sigma"] = (
            torch.eye(num_features)
            .unsqueeze(0)
            .repeat((self.num_clusters, 1, 1))
            .to(self.device)
        )

        self.parameters["pi"] = (
            torch.ones((self.num_clusters)).to(self.device) / self.num_clusters
        )

    def _e_step(self, x):
        likelihood = torch.zeros((x.shape[0], self.num_clusters)).to(x.device)
        for k in range(self.num_clusters):
            gaussian = torch.distributions.MultivariateNormal(
                self.parameters["mu"][k], self.parameters["sigma"][k]
            )
            likelihood[:, k] = self.parameters["pi"][k] * torch.exp(
                gaussian.log_prob(x)
            )

        gammma = likelihood / likelihood.sum(axis=1).unsqueeze(1)
        return gammma

    def _m_step(self, x, gamma):
        gamma_sum = gamma.sum(axis=0)
        self.parameters["pi"] = gamma_sum / x.shape[0]

        self.parameters["mu"] = gamma.T @ x / gamma_sum.unsqueeze(1)

        sigma = torch.zeros_like(self.parameters["sigma"])

        for k in range(self.num_clusters):
            x_centered = x - self.parameters["mu"][k]
            sigma[k] = ((gamma[:, k] * x_centered.T) @ x_centered) / gamma_sum[k]

    @torch.no_grad()
    def fit(self, x):
        x = x.float().to(self.device)

        self._initialise_parameters(x)
        log_likelihood = None

        logs_over_time = []

        for iter in range(self.max_iter):
            responsibilities = self._e_step(x)
            self._m_step(x, responsibilities)
            log_likelihood_new = responsibilities.sum()

            if (
                iter > 0
                and torch.sqrt((log_likelihood - log_likelihood_new) ** 2) < self.tol
            ):
                break

            log_likelihood = log_likelihood_new

            logs_over_time.append(log_likelihood)

    @torch.no_grad()
    def fit_predict(self, x):
        self.fit(x)
        labels = self.predict(x)
        return labels

    @torch.no_grad()
    def predict(self, x):
        responsibilities = self._e_step(x)
        return responsibilities.argmax(axis=1)
