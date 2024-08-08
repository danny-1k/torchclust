import unittest
import torch
from sklearn.datasets import make_blobs

from torchclust import density


class TestDensityBasedAlgorithms(unittest.TestCase):
    def test_dbscan(self):
        x = torch.randn(1000, 3)
        dbscan = density.DBSCAN(min_samples=10, eps=0.1)
        labels = dbscan.fit_predict(x)

        self.assertEqual(labels.shape[0], x.shape[0])

    def test_dbscan_clustering(self):
        x, labels = make_blobs(1000, n_features=3, centers=3)
        dbscan = density.DBSCAN(min_samples=10, eps=0.1)
        labels_ = dbscan.fit_predict(torch.from_numpy(x))

        self.assertEqual(labels.shape[0], labels_.shape[0])
