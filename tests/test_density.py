import unittest
import torch
from sklearn.datasets import make_blobs

from torchclust import density


class TestDensityBasedAlgorithms(unittest.TestCase):
    def setUp(self):
        self.x, self.labels = make_blobs(1000, n_features=3, centers=3)
        self.x = torch.from_numpy(self.x)
        self.labels = torch.from_numpy(self.labels)

    def test_dbscan(self):
        dbscan = density.DBSCAN(min_samples=10, eps=0.1)
        labels = dbscan.fit_predict(self.x)
        self.assertEqual(labels.shape[0], self.x.shape[0])