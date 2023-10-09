import os
import random
import typing
import datetime
import pickle

import pandas as pd
import numpy as np

from . import abstract_model_test

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from custom_metrics import (
    timing,
    storage
)

class SklearnModelTest(abstract_model_test.AbstractModelTest):
    def __init__(self, model, model_specs_dict: typing.Dict):
        self._model = model
        self._presaved_models_paths_dict = model_specs_dict["presaved_paths"]
        self._evaluation_metrics = []

        self._run_id = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_sklearn"

        self._metrics_output_path = f"{model_specs_dict['model_specs']['paths']['metrics_output_path']}/{self._run_id}"
        if not os.path.exists(self._metrics_output_path):
            os.makedirs(self._metrics_output_path)
            print("Metrics output directory created successfully")

    def __seed_all(self, seed):
        if not seed:
            seed = 10

        print("[ Using Seed : ", seed, " ]")

        np.random.seed(seed)
        random.seed(seed)


    def __validate_model(self, X, y_true):
        y_pred = self._model.predict(X)
        y_pred_prob = self._model.predict_proba(X)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_prob[:, 1])

        dummy_data = X[0].reshape(1, -1)
        inference_time = timing.sklearn_inference_time(self._model, dummy_data)

        # TODO: Change this to be only used in case model is random forest
        model_size = storage.sklearn_random_forest_compute_model_size_mb(self._model._model)

        return [acc, f1, prec, recall, roc_auc, inference_time, model_size]


    def execute(self, data):
        # Reset all seed to ensure reproducibility
        self.__seed_all(0)

        # Get item from train data
        X_test = [item[0] for item in data]
        y_test = [item[1] for item in data]

        for fold_index in self._presaved_models_paths_dict.keys():
            print('------------fold no---------{}----------------------'.format(fold_index))

            # Load the current fold model
            model_filename = f"{self._presaved_models_paths_dict[fold_index]}"
            self._model = pickle.load(open(model_filename, 'rb'))

            # Get the train metrics
            test_metrics = self.__validate_model(X_test, y_test)
            test_metrics = [fold, *test_metrics]

            # Append the metrics to be further exported
            self._metrics_list.append(test_metrics)

            self._model.reset()

        metrics_df = pd.DataFrame(self._metrics_list, columns=["fold", "acc", "f1", "prec", "recall", "roc_auc", "inference_time", "model_size"])
        metrics_df.to_csv(f"{self._metrics_output_path}/test_metrics_sklearn_{self._model_name}.csv")
