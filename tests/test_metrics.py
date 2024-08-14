import unittest
import torch


from torchclust import metrics


class TestExternalMetrics(unittest.TestCase):
    def setUp(self):
        self.labels = torch.Tensor([1, 0, 2, 5, 2, 3, 2, 3, 3, 4])
        self.pred = torch.Tensor([4, 5, 4, 1, 3, 0, 5, 3, 5, 4])

    def test_rand_index(self):
        ri = metrics.rand_index(self.labels, self.pred)
        self.assertAlmostEqual(round(ri, 4), 0.7111)

    def test_mutual_information(self):
        mi = metrics.mutual_information(self.labels, self.pred)
        self.assertAlmostEqual(round(mi, 4), 0.8456)

    def test_purity_score(self):
        purity = metrics.purity_score(self.labels, self.pred)
        self.assertAlmostEqual(round(purity, 4), 0.5000)
