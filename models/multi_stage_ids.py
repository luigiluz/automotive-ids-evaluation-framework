import torch
import sklearn

import numpy as np
import pandas as pd

from torch import nn

from . import conv_net_ids
from . import sklearn_classifier

class MultiStageIDS():
    def __init__(self, cnn_model, rf_model, device):
        super(MultiStageIDS, self).__init__()
        self._cnn_model = cnn_model
        self._rf_model = rf_model
        self._device = device

        # TODO: adicionar o carregamento dos modelos ja existentes

    def __multi_stage_ensemble(self, x_rf, x_dl, weight=0.5):
        # definir como isso vai acontecer
        ms_output = x_rf + x_dl
        return ms_output

    def forward(self, x):
        # cnn layers
        x = self._cnn_model.feature_extraction_layer(x)
        x = torch.flatten(x, 1)

        # possible place for random forest model
        # TODO: replace this for the previous layer output
        rf_input = np.random.normal(0, 0.1, (64, 7))
        x_rf = torch.from_numpy(self._rf_model.predict(rf_input)).reshape(-1, 1).to(self._device)

        x = self._cnn_model.binary_classification_layer(x)
        # this is the entrypoint for the first fc layer (20k+ units for regular cnn, 8291 for pruned cnn)
        # x = self._cnn_model.binary_classification_layer_fc1(x)

        # # this is the second fc layer (64 units)
        # x = self._cnn_model.binary_classification_layer_fc2(x)
        x_dl = torch.sigmoid(x)

        # as entradas sao dois vetores de mesmo n por m
        x_ms = self.__multi_stage_ensemble(x_rf, x_dl)

        output = {
            "rf": x_rf,
            "dl": x_dl,
            "ms": x_ms
        }

        return output
