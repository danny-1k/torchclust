import torch


def make_blobs(
    num_samples: int = 100,
    num_features: int = 2,
    num_centers: int = 3,
    cluster_std: float = 1.0,
    center_box=(-10, 10),
    random_state: int = None,
    device: str = "cpu",
):
    """
    Isotropic Gaussian Blobs dummy dataset for clustering.

    Args:
        num_samples (int, optional): Number of samples. Defaults to 100.
        num_features (int, optional): Number of features. Defaults to 2.
        num_centers (int, optional): Number of cluster centers. Defaults to 3.
        cluster_std (float, optional): Cluster spread. Defaults to 1.0.
        center_box (tuple, optional): Bounding box for each cluster centroid. Defaults to (-10, 10).
        random_state (int, optional): Random state for reproducibility. Defaults to None.
        device (str, optional): Device, e.g cuda.
    """

    if random_state is not None:
        torch.manual_seed(random_state)

    centers = torch.zeros((num_centers, num_features)).uniform_(*center_box).to(device)
    n_samples_per_center = int(num_samples // num_centers)
    num_remainder = num_samples - (n_samples_per_center * num_centers)

    x = torch.randn((num_samples, num_features)).to(device)
    y = torch.zeros(num_samples).long().to(device)

    for k in range(num_centers):
        if k == 0:
            x[: n_samples_per_center + num_remainder] = (
                x[: n_samples_per_center + num_remainder] * cluster_std + centers[k]
            )

            y[: n_samples_per_center + num_remainder] = k

        else:
            x[
                k * n_samples_per_center
                + num_remainder : k * n_samples_per_center
                + num_remainder
                + n_samples_per_center
            ] = (
                x[
                    k * n_samples_per_center
                    + num_remainder : k * n_samples_per_center
                    + num_remainder
                    + n_samples_per_center
                ]
                * cluster_std
                + centers[k]
            )

            y[
                k * n_samples_per_center
                + num_remainder : k * n_samples_per_center
                + num_remainder
                + n_samples_per_center
            ] = k

    return x, y
