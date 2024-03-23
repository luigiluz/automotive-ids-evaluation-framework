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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from custom_metrics import (
    timing,
    storage
)

class SklearnModelTest(abstract_model_test.AbstractModelTest):
    def __init__(self, model, model_specs_dict: typing.Dict):
        self._model = model
        self._model_name = model_specs_dict['model_specs']['model_name']
        self._presaved_models_paths_dict = model_specs_dict["model_specs"]["presaved_paths"]
        self._metrics_list = []
        self._labeling_schema = model_specs_dict['feat_gen']['config']['labeling_schema']

        self._run_id = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_sklearn_test"

        # TODO: Get this from json config file
        art_path = "/home/lfml/workspace/artifacts"
        self._artifacts_path = f"{art_path}/{self._run_id}"

        if not os.path.exists(self._artifacts_path):
            os.makedirs(self._artifacts_path)
            print("Artifacts output directory created successfully")

        self._metrics_output_path = f"{self._artifacts_path}/metrics"
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

        # dummy_data = X[0].reshape(1, -1)
        dummy_data = np.random.rand(64, 116)
        inference_time = timing.sklearn_inference_time(self._model, dummy_data)
        inference_time = inference_time / len(dummy_data)

        # TODO: Change this to be only used in case model is random forest
        # model_size = storage.sklearn_random_forest_compute_model_size_mb(self._model._model)

        conf_matrix = confusion_matrix(y_true, y_pred)

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob[:, 1], drop_intermediate=True)

        self._confusion_matrix = conf_matrix
        self._roc_metrics = np.concatenate((fpr.reshape(-1, 1), tpr.reshape(-1, 1), thresholds.reshape(-1, 1)), axis=1)

        return [acc, f1, prec, recall, roc_auc, inference_time]


    def execute(self, data):
        # Reset all seed to ensure reproducibility
        self.__seed_all(0)

        # Get item from train data
        X_test_full = [item[0] for item in data]
        y_test_full = [item[1] for item in data]

        unique_y_values = np.unique(y_test_full)
        # 0 is equal to the normal label, > 0 is equal to attack (in TOW-IDS)
        existing_labels = unique_y_values[unique_y_values > 0]

        for label in existing_labels:
            print(f"Current selected label: {label}")
            normal_and_label_entries_cond = (y_test_full == 0) | (y_test_full == label)

            y_test = y_test_full[normal_and_label_entries_cond]
            X_test = X_test_full[normal_and_label_entries_cond]

            attack_suffix = f"attack_{label}"

            for fold_index in self._presaved_models_paths_dict.keys():
                print('------------fold no---------{}----------------------'.format(fold_index))

                # Load the current fold model
                model_filename = f"{self._presaved_models_paths_dict[fold_index]}"
                self._model = pickle.load(open(model_filename, 'rb'))

                # Get the train metrics
                test_metrics = self.__validate_model(X_test, y_test)
                test_metrics = [fold_index, *test_metrics]

                # Append the metrics to be further exported
                self._metrics_list.append(test_metrics)

                self._model.reset()

                metrics_df = pd.DataFrame(self._metrics_list, columns=["fold", "acc", "f1", "prec", "recall", "roc_auc", "inference_time"])
                metrics_df.to_csv(f"{self._metrics_output_path}/{attack_suffix}_test_metrics_sklearn_{self._model_name}.csv")
                confusion_matrix_df = pd.DataFrame(self._confusion_matrix)
                confusion_matrix_df.to_csv(f"{self._metrics_output_path}/{attack_suffix}_confusion_matrix_{self._labeling_schema}_fold_{fold_index}_{self._model_name}.csv")
                roc_metrics_df = pd.DataFrame(self._roc_metrics, columns=["fpr", "tpr", "thresholds"])
                roc_metrics_df.to_csv(f"{self._metrics_output_path}/{attack_suffix}_roc_metrics_{self._labeling_schema}_fold_{fold_index}_{self._model_name}.csv")
