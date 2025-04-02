import wandb
import pytorch_lightning as pl
import logging
import pytorch_lightning as pl
import torch
import torchmetrics as tm
import torch.nn.functional as F
import torch.nn as nn

from pathlib import Path
from einops import rearrange
from typing import Any, Dict, List, Tuple, Type, Union
from torch import Tensor
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from byol.paths import Path_Handler
from byol.config import load_config, update_config, load_config_finetune
from byol.models import BYOL
from byol.datamodules import RGZ_DataModule_Finetune

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.linear(x)
        return x


class MLPHead(nn.Module):
    """
    Fully connected head with a single hidden layer. Batchnorm applied as first layer so that
    feature space of encoder doesn't need to be normalized.
    """

    def __init__(self, input_dim, depth, width, output_dim):
        super(MLPHead, self).__init__()

        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(depth):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(width, width),
                    nn.GELU(),
                )
            )

        self.output_layer = nn.Sequential(
            nn.Linear(width, output_dim),
        )

    def forward(self, x):
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)

        return x

''' MRL functions modified from https://github.com/RAIVNLab/MRL/tree/main '''

#modifications to code for self.efficient = True not complete yet

class MRL_Linear_Layer(nn.Module):
    #nesting_liat[i] = input_dim, num_classes = output_dim 
    def __init__(self, nesting_list: List, num_classes=1000, efficient=False, **kwargs):
        super(MRL_Linear_Layer, self).__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes # Number of classes for classification
        self.efficient = efficient
        if self.efficient:
            # setattr(self, f"nesting_classifier_bn{0}", nn.Linear(nesting_list[-1], **kwargs))
            setattr(self, f"nesting_classifier_{0}", nn.Linear(nesting_list[-1], self.num_classes, **kwargs))        
        else:    
            for i, num_feat in enumerate(self.nesting_list):
                # setattr(self, f"nesting_classifier_bn{i}", nn.BatchNorm1d(num_feat, **kwargs))
                setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))    

    def reset_parameters(self):
        if self.efficient:
            # self.nesting_classifier_bn0.reset_parameters()
            self.nesting_classifier_0.reset_parameters()
        else:
            for i in range(len(self.nesting_list)):
                # getattr(self, f"nesting_classifier_bn{i}").reset_parameters()
                getattr(self, f"nesting_classifier_{i}").reset_parameters()


    def forward(self, x):
        nesting_logits = ()
        for i, num_feat in enumerate(self.nesting_list):
            if self.efficient:
                if self.nesting_classifier_0.bias is None:
                    # x = 
                    nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()), )
                else:
                    nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias, )
            else:
                # x =  getattr(self, f"nesting_classifier_bn{i}")(x[:, :num_feat])
                nesting_logits +=  (getattr(self, f"nesting_classifier_{i}")(x[vv:, :num_feat]),)

        return nesting_logits
  
        
class Matryoshka_CE_Loss(nn.Module):
    def __init__(self, relative_importance: List[float]=None, **kwargs):
        super(Matryoshka_CE_Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)
        # relative importance shape: [G]
        self.relative_importance = relative_importance

    def forward(self, output, target):
        # output shape: [G granularities, N batch size, C number of classes]
        # target shape: [N batch size]

        # Calculate losses for each output and stack them. This is still O(N)
        losses = torch.stack([self.criterion(output_i, target) for output_i in output])
        
        # Set relative_importance to 1 if not specified
        rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance)
        
        # Apply relative importance weights
        weighted_losses = rel_importance * losses
        return weighted_losses.sum()
 