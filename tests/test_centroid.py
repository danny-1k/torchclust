import unittest
import torch
from sklearn.datasets import make_blobs


from torchclust import centroid


class TestCentroidBasedAlgorithms(unittest.TestCase):
    def test_kmeans(self):
        x = torch.randn(1000, 3)
        kmeans = centroid.KMeans(num_clusters=5)
        labels = kmeans.fit_predict(x)

        self.assertEqual(labels.shape[0], x.shape[0])

    def test_kmeans_clustering(self):
        x, labels = make_blobs(1000, n_features=3, centers=3)
        dbscan = centroid.KMeans(num_clusters=3)
        labels_ = dbscan.fit_predict(torch.from_numpy(x))

        self.assertEqual(labels.shape[0], labels_.shape[0])
