import torch
import sklearn
import pickle

import numpy as np
import pandas as pd

from torch import nn

from . import conv_net_ids
from . import pruned_conv_net_ids
from . import sklearn_classifier

class MultiStageIDS(nn.Module):
    def __init__(self, ensemble_inputs=2):
        super(MultiStageIDS, self).__init__()
        self._pruned_cnn_model = None
        self._rf_model = None

        self.mlp_ensemble_layer = nn.Sequential(
            nn.Linear(in_features=ensemble_inputs, out_features=1),
        )

        # TODO: depois remover isso pra nao ficar hardcoded
        self._device = torch.device("cuda:1")

    def load_stages_models(self, rf_path: str, cnn_path: str):
        # Load presaved models
        ## Load first stage model
        self._rf_model = pickle.load(open(rf_path, 'rb'))

        ## Load second stage model
        self._pruned_cnn_model = pruned_conv_net_ids.PrunedConvNetIDS()
        self._pruned_cnn_model.load_state_dict(torch.load(cnn_path, map_location='cpu'))

    def forward_first_stage(self, x):
        # First stage
        # This consider the input is in the shape
        # (64, 1, 44, 116)
        # Axis 2 correspond to the 44, that is the number of grouped packets
        x_rf = torch.sum(x, axis=2)
        x_rf = x_rf.reshape(-1, x_rf.shape[-1])
        x_rf = x_rf.cpu()
        y1 = torch.tensor(self._rf_model.predict(x_rf))
        y1 = y1.reshape(y1.shape[-1], -1)

        return y1

    def forward_second_stage(self, x):
        # Second stage
        y2 = self._pruned_cnn_model(x)

        return y2

    def forward(self, x):
        # Ensemble stage
        # Combine the outputs into an tensor with shape (64, (y1_cols + y2_cols))
        y1 = self.forward_first_stage(x).to(self._device)
        y2 = self.forward_second_stage(x).to(self._device)

        # print(f"y1 = {y1}")
        # print(f"y1.shape = {y1.shape}")
        # print(f"y2 = {y2}")
        # print(f"y2.shape = {y2.shape}")

        x = torch.cat((y1, y2), axis=1)

        # print(f"x = {x}")
        # print(f"x.shape = {x.shape}")

        x = self.mlp_ensemble_layer(x)
        x = torch.sigmoid(x)

        return x
