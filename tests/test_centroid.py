import unittest
import torch

from torchclust import centroid


class TestCentroidBasedAlgorithms(unittest.TestCase):
    def test_kmeans(self):
        x = torch.randn(1000, 3)
        kmeans = centroid.KMeans(num_clusters=5)
        labels = kmeans.fit_predict(x)

        self.assertEqual(labels.shape[0], x.shape[0])
