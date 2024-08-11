![](./assets/logo.png)

# Torchclust: Clustering Algorithms written with Pytorch for running on GPU

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
        - Calinski-Harabasz Score / Variance Ratio Criterion
    - External
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

*Be sure to make to make sure the GPU version of pytorch is installed if you intend to run the algorithms on GPU.*

```bash
pip install torchclust
```
## Usage

#### Kmeans on sklearn make_blobs
```python
import torch
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from torchclust.centroid import KMeans

x, _ = make_blobs(1000, n_features=3, centers=3)
x = torch.from_numpy(x)

kmeans = KMeans(num_clusters=3)
labels = kmeans.fit_predict(x)

plt.scatter(x[:, 0], x[:, 1], c=labels)
plt.show()
```
