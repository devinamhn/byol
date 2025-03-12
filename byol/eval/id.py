import torch
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
# from umap import UMAP
import numpy as np
from sklearn.neighbors import NearestNeighbors
import skdim
from byol.eval.TwoNN import TwoNN as myTwoNN
import scipy.spatial.distance as dist
# # HPARAMS
PCA_COMPONENTS = 200
UMAP_N_NEIGHBOURS = 75
UMAP_MIN_DIST = 0.01
METRIC = "cosine"
seed = 69


def calculate_intrinsic_dimension(data):
    """
    Calculates the intrinsic dimension of a dataset using the 'Two Nearest Neighbors' method 
    from the scikit-dimension package.
    
    https://scikit-dimension.readthedocs.io/en/latest/skdim.id.TwoNN.html
    
    Args:
        data: A numpy array representing the dataset.

    Returns:
        float: The estimated intrinsic dimension. 
    """
    
    # Initialize the dimensionality reduction object
    
    # dr = skdim.id.TwoNN() 
    # dr = myTwoNN(discard_fraction= 0.15) 
    dr = skdim.id.lPCA() 
    
    # Fit the model to the data
    dr.fit(data) 
    
    # Get the estimated intrinsic dimension
    intrinsic_dim = dr.dimension_
    
    return intrinsic_dim

test_features = torch.load('test_features_rgz_beforeft.pt')
test_features_ft = torch.load('test_features_rgz_afterft.pt')

pdist_rgz = dist.pdist(test_features, 'cosine')
sqform = dist.squareform(pdist_rgz)

# plt.imshow(sqform, interpolation='nearest')
# plt.savefig('heatmap.png')

discard_fraction = 0.7

# twonn = skdim.id.TwoNN(discard_fraction) 
twonn = myTwoNN(discard_fraction) 

twonn.fit(test_features)

intrinsic_dim_twonn = twonn.dimension_
print('estimated id', intrinsic_dim_twonn)

X = twonn.x_
y = twonn.y_


# twonnft = skdim.id.TwoNN(discard_fraction) 
twonnft = myTwoNN(discard_fraction) 

twonnft.fit(test_features_ft)

intrinsic_dim_twonnft = twonnft.dimension_
print('estimated id', intrinsic_dim_twonnft)

Xft = twonnft.x_
yft = twonnft.y_

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle(f'iD estimation for RGZ Embeddings with 2NN f ={discard_fraction}')
ax1.plot(X, y)
ax1.set_title(f'before ft id {round(intrinsic_dim_twonn, 4)}')
ax2.plot(Xft, yft)
ax2.set_title(f'after ft id {round(intrinsic_dim_twonnft, 4)}')
plt.savefig(f'mus_fmu_cosine{discard_fraction}.png')


