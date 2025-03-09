import torch
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from einops import pack
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from umap import UMAP
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

from byol.utilities import embed_dataset
from byol.datamodules import RGZ108k
from byol.datamodules import MBFRFull, MBFRConfident, MBFRUncertain
from byol.models import BYOL
from byol.paths import Path_Handler
from byol.resnet import ResNet, BasicBlock
from byol.finetuning import FineTune
from byol.config import load_config_finetune

# # HPARAMS
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
        device = next(encoder.parameters()).device
        feature_bank = []
        target_bank = []
        for data in tqdm(train_loader):
            # Load data and move to correct device
            x, y = data

            x_enc = encoder(x.to(device))

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

paths = Path_Handler()._dict()

#load pretrained byol checkpoint
byol = BYOL.load_from_checkpoint(paths["main"] / "byol.ckpt")

#load finetuned byol checkpoint
# finetuned_ckpt_path = 'model_finetune'
finetuned_ckpt_path = paths["main"] / "files/finetune/hbo9tznc/BYOL_finetune_reproduce/hbo9tznc/checkpoints/epoch=299-step=3600.ckpt"
finetuned_ckpt = torch.load(finetuned_ckpt_path)

# get state dict keys
# state_dict = finetuned_ckpt['state_dict'].keys()

#get encoder and head parameters
encoder_weights = {k: v for k, v in finetuned_ckpt["state_dict"].items() if k.startswith("encoder.")}
head_weights = {k: v for k, v in finetuned_ckpt["state_dict"].items() if k.startswith("head.")}

''' 
rename encoder weights - imp for loading the finetuned parameters from the ckpt since
finetuned model's state dict keys begin with 'encoder' and won't match with
the state dict of a new FineTune model instance
'''
encoder_weights_renamed = {k.replace('encoder.', ''): v for k,v in encoder_weights.items()}

# byol = FineTune.load_from_checkpoint(encoder_weights, head_weights)
# byol = FineTune.load_from_checkpoint(finetuned_ckpt_path)

config_finetune = load_config_finetune()
#create an object of the finetuning class
head = 'linear' 
config_finetune["finetune"]["seed"] = seed

# model = FineTune(byol.encoder, head, dim = byol.encoder.dim, )
model = FineTune(
    byol.encoder,
    head,
    dim=byol.encoder.dim,
    n_classes= config_finetune["finetune"]["n_classes"],
    n_epochs= config_finetune["finetune"]["n_epochs"],
    n_layers= config_finetune["finetune"]["n_layers"],
    batch_size= config_finetune["finetune"]["batch_size"],
    lr_decay= config_finetune["finetune"]["lr_decay"],
    seed= config_finetune["finetune"]["seed"],
    head_type= config_finetune["finetune"]["head"],
)
#the model object should have updated layer names!!
# print('from ckpt', (encoder_weights.keys()))


# print('new keys', encoder_weights_renamed.keys())


# print('from model',(model.encoder.state_dict().keys()))

#should also work if I directly use byol.encoder
encoder = model.encoder
encoder.load_state_dict(encoder_weights_renamed)
# head.load_state_dict(head_weights)
# head = model.head

transform = T.Compose(
    [
        T.CenterCrop(70),
        T.ToTensor(),
        T.Normalize((0.008008896,), (0.05303395,)),
    ]
)

rgz = RGZ108k(
    paths["rgz"],
    train=True,
    transform=transform,
    download=False,
    remove_duplicates=False,
    cut_threshold=25,
    mb_cut=True,
)

reducer = Reducer(encoder)
reducer.fit(rgz)
X_umap = reducer.transform(rgz)


alpha = 0.6
marker_size = 0.1
fig_size = (10 / 3, 3)
fontsize = 9
marker = "o"

fig, ax = plt.subplots()
fig.set_size_inches(fig_size)
scatter = ax.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=reducer.targets,
    cmap="Spectral",
    s=marker_size,
    marker=marker,
    vmin=25,
    vmax=100,
    alpha=alpha,
)

fig.savefig("byol_umap_rgz_finetuned.png", bbox_inches="tight", pad_inches=0.05, dpi=600)
