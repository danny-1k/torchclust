![](./assets/logo.png)

# Torchclust: Clustering Algorithms written with Pytorch for running on GPU
![License](https://img.shields.io/github/license/hmunachi/nanodl?style=flat-square) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/daniel-ik-human) [![Twitter](https://img.shields.io/twitter/follow/1sn00s?style=social)](https://twitter.com/1sn00s)

Torchclust was developed to solve the issue of having to convert Pytorch Tensors to Numpy arrays and moving them to the CPU from the GPU in order to utilise frameworks such as scikit-learn.

Torchclust features implementations of common clustering algorithms with a scikit-learn feel.

## Implemented algorithms
- Centroid-based Clustering
    - KMeans
    - MeanShift
- Density-based Clustering
    - DBSCAN
    - Gaussian Mixture Model
- Deep / Learning-based Clustering
    - Self-Organising Maps
- Metrics
    - Internal
        - Silhouette Score
        - Interia
        - Davies-Bouldin Index 
        - Calinski-Harabasz Score / Variance Ratio Criterion
    - External
        - Purity Score
        - Rand Index
        - Adjusted Rand Index
        - Mutual Information
        - Normalised Mutual Information

## Contributing
This is still an ongoing project and contributions from the opensource community are warmly welcomed.

Contributions can be made in various forms:
- Writing docs / Updating README
- Fixings bugs
- More efficient implementations of algorithnms
- Or even implementing more algorithms

## Installation

*Be sure the GPU version of pytorch is installed if you intend to run the algorithms on GPU.*

```bash
pip install torchclust
```
## Usage

#### Kmeans on gaussian blobs
```python
import torch
import matplotlib.pyplot as plt

from torchclust.utils.datasets import make_blobs
from torchclust.centroid import KMeans

x, _ = make_blobs(1000, num_features=2, centers=3)
x = torch.from_numpy(x)

kmeans = KMeans(num_clusters=3)
labels = kmeans.fit_predict(x)

plt.scatter(x[:, 0], x[:, 1], c=labels)
plt.show()
```
