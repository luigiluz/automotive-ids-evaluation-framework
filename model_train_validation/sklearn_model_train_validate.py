import datetime
import typing
import pickle
import json
import os
import random

import pandas as pd
import numpy as np

from . import abstract_model_train_validate

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

# from numba import jit, cuda

from custom_metrics import timing, storage

class SklearnModelTrainValidation(abstract_model_train_validate.AbstractModelTrainValidate):
    def __init__(self, model, model_config_dict: typing.Dict):
        self._model = model
        self._model_name = model_config_dict["model_name"]
        self._metrics_list = []
        self._run_id = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_sklearn_train_val"
        self._hyperparameters_grid = model_config_dict["hyperparams_grid"]

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

        self._models_output_path = f"{self._artifacts_path}/models"
        if not os.path.exists(self._models_output_path):
            os.makedirs(self._models_output_path)
            print("Models output directory created successfully")


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
        # model_size = storage.sklearn_random_forest_compute_model_size_mb(self._model._model['clf'])

        return [acc, f1, prec, recall, roc_auc, inference_time]

    # @jit(target_backend='cuda')
    def execute(self, train_data):
        # Reset all seed to ensure reproducibility
        self.__seed_all(0)

        # Get item from train data
        X = [item[0] for item in train_data]
        X = np.array(X)
        y = [item[1] for item in train_data]
        y = np.array(y)

        print(">> Execute")
        print(f">> X.shape = {X.shape}")
        print(f">> y.shape = {y.shape}")

        # Create the stratified KFold object
        skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

        ## Used only for random search cross validation
        # # Create random search cross validation
        # random_search_cv = RandomizedSearchCV(estimator = self._model._model,
        #     param_distributions = self._hyperparameters_grid,
        #     scoring = 'f1',
        #     n_iter = 30,
        #     cv = skf,
        #     verbose = 2,
        #     random_state = 42)

        # search = random_search_cv.fit(X, y)

        # cv_results = search.cv_results_
        # best_estimator = search.best_estimator_

        # model_filename = f"random_search_best_{self._model_name}.pkl"
        # with open(f"{self._models_output_path}/{model_filename}", "wb") as modelfile:
        #     pickle.dump(best_estimator, modelfile)

        # cv_results_filename = f"random_search_cv_results_{self._model_name}.csv"
        # cv_results_df = pd.DataFrame(cv_results)
        # cv_results_df.to_csv(f"{self._metrics_output_path}/{cv_results_filename}")

        # print(">> Proceeding to evaluate the best estimator in the validation set")
        # # Get the best estimator
        # self._model._model = best_estimator

        # ## Evaluate the best model
        # for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        #     # Get the test data
        #     X_test = X[test_idx]
        #     y_test = y[test_idx]
        #     # Test (validate) the model
        #     test_metrics = self.__validate_model(X_test, y_test)
        #     test_metrics = ["validation", fold, *test_metrics]
        #     self._metrics_list.append(test_metrics)

        # metrics_df = pd.DataFrame(self._metrics_list, columns=["step", "fold", "acc", "f1", "prec", "recall", "roc_auc", "inference_time", "model_size"])
        # metrics_df.to_csv(f"{self._metrics_output_path}/train_val_metrics_{self._model_name}.csv")

        # Used to generate a model for each fold
        # TODO: Find a better way to do this validation
        # This is to check if y has more than one dimension, for the multiclass case to work with skf.split
        try:
            y = np.array(y).argmax(1)
        except:
            pass

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print('------------fold no---------{}----------------------'.format(fold))

            # Select the train data
            X_train = X[train_idx]
            y_train = y[train_idx]

            # Select the test data
            X_test = X[test_idx]
            y_test = y[test_idx]

            # Train the model
            self._model.train(X_train, y_train)

            # Get the train metrics
            train_metrics = self.__validate_model(X_train, y_train)
            train_metrics = ["train", fold, *train_metrics]

            # Test (validate) the model
            test_metrics = self.__validate_model(X_test, y_test)
            test_metrics = ["validation", fold, *test_metrics]

            # Append the metrics to be further exported
            self._metrics_list.append(train_metrics)
            self._metrics_list.append(test_metrics)
            # Export the current fold model
            model_filename = f"{self._model_name}_fold_{fold}.pkl"
            with open(f"{self._models_output_path}/{model_filename}", "wb") as file:
                pickle.dump(self._model, file)

            self._model.reset()

        metrics_df = pd.DataFrame(self._metrics_list, columns=["step", "fold", "acc", "f1", "prec", "recall", "roc_auc", "inference_time"])
        metrics_df.to_csv(f"{self._metrics_output_path}/train_val_metrics_{self._model_name}.csv")
