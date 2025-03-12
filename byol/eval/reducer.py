import torch
import numpy as np

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from skdim.id import TwoNN
from einops import pack
from torch.utils.data import DataLoader
from tqdm import tqdm
from umap import UMAP

PCA_COMPONENTS = 200
UMAP_N_NEIGHBOURS = 75
UMAP_MIN_DIST = 0.01
METRIC = "cosine"
seed = 69

class Reducer:
    def __init__(self, encoder):
        self.encoder = encoder
        self.pca = PCA(n_components=PCA_COMPONENTS, random_state=seed)
        self.umap = UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBOURS,
            min_dist=UMAP_MIN_DIST,
            metric="cosine",
            random_state=seed,
        )

    def embed_dataset(self, data, batch_size=400):
        train_loader = DataLoader(data, batch_size, shuffle=False)
        device = next(self.encoder.parameters()).device
        feature_bank = []
        target_bank = []
        for data in tqdm(train_loader):
            # Load data and move to correct device
            x, y = data

            x_enc = self.encoder(x.to(device))

            feature_bank.append(x_enc.squeeze().detach().cpu())
            # target_bank.append(y.to(device).detach().cpu())

        # Save full feature bank for validation epoch
        features = torch.cat(feature_bank)
        targets = np.ones(features.shape[0])
        # targets = torch.cat(target_bank)


        return features, targets

    def fit(self, data):
        print("Fitting reducer")
        features, targets = self.embed_dataset(data)
        self.features = features
        self.targets = targets

        self.pca.fit(self.features)
        self.umap.fit(self.pca.transform(self.features))

    def transform(self, data):
        # x = self.encoder(x.cuda()).squeeze().detach().cpu().numpy()
        x, _ = self.embed_dataset(data)
        x = self.pca.transform(x)
        x = self.umap.transform(x)
        return x

    def transform_pca(self, data):
        x, _ = self.embed_dataset(data)
        x = self.pca.transform(x)
        return x


class TwoNNReducer:
    def __init__(self, encoder):
        self.encoder = encoder
        self.twoNN = TwoNN(discard_fraction = 0.01)

    def embed_dataset(self, data, batch_size=400):
        train_loader = DataLoader(data, batch_size, shuffle=False)
        device = next(self.encoder.parameters()).device
        feature_bank = []
        target_bank = []
        for data in tqdm(train_loader):
            # Load data and move to correct device
            x, y = data

            x_enc = self.encoder(x.to(device))

            feature_bank.append(x_enc.squeeze().detach().cpu())
            # target_bank.append(y.to(device).detach().cpu())

        # Save full feature bank for validation epoch
        features = torch.cat(feature_bank)
        targets = np.ones(features.shape[0])
        # targets = torch.cat(target_bank)


        return features, targets

    def fit(self, data):
        print("Fitting reducer")
        features, targets = self.embed_dataset(data)
        self.features = features
        self.targets = targets
        print(self.features.shape)
        self.twoNN.fit(self.features)
        print('est id', self.twoNN.dimension_)
        # self.pca.fit(self.features)
        # self.umap.fit(self.pca.transform(self.features))

    def transform(self, data):
        # x = self.encoder(x.cuda()).squeeze().detach().cpu().numpy()
        x, _ = self.embed_dataset(data)
        x = self.twoNN.transform(x)
        # x = self.pca.transform(x)
        # x = self.umap.transform(x)
        return x

    def transform_pca(self, data):
        x, _ = self.embed_dataset(data)
        x = self.pca.transform(x)
        return x

    def transform_2NN(self, data):
        x, _ = self.embed_dataset(data)
        x = self.twoNN.transform(x)
        return x