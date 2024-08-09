import unittest
import torch
from sklearn.datasets import make_blobs


from torchclust import centroid


class TestCentroidBasedAlgorithms(unittest.TestCase):
    def setUp(self):
        self.x, self.labels = make_blobs(1000, n_features=3, centers=3)
        self.x = torch.from_numpy(self.x)
        self.labels = torch.from_numpy(self.labels)

    def test_kmeans(self):
        kmeans = centroid.KMeans(num_clusters=3)
        labels = kmeans.fit_predict(self.x)

        self.assertEqual(labels.shape[0], self.x.shape[0])

    def test_meanshift(self):
        meanshift = centroid.MeanShift(band_width=2)
        labels = meanshift.fit_predict(self.x)

        self.assertEqual(labels.shape[0], self.x.shape[0])
