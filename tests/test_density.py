import unittest
import torch

from torchclust import density


class TestDensityBasedAlgorithms(unittest.TestCase):
    def test_dbscan(self):
        x = torch.randn(1000, 3)
        dbscan = density.DBSCAN(min_samples=10, eps=0.1)
        labels = dbscan.fit_predict(x)

        self.assertEqual(labels.shape[0], x.shape[0])
