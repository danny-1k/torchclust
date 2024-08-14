import unittest
import torch
from torchclust.utils.datasets import make_blobs
from torchclust import centroid


class TestCentroidBasedAlgorithms(unittest.TestCase):
    def setUp(self):
        self.x, self.labels = make_blobs(
            1000, num_features=3, num_centers=3, random_state=42
        )

    def test_kmeans(self):
        kmeans = centroid.KMeans(num_clusters=3)
        labels = kmeans.fit_predict(self.x)

        self.assertEqual(labels.shape[0], self.x.shape[0])

    def test_meanshift(self):
        meanshift = centroid.MeanShift(band_width=2)
        labels = meanshift.fit_predict(self.x)

        self.assertEqual(labels.shape[0], self.x.shape[0])
