import os
import random
import typing
import datetime

import pandas as pd
import numpy as np

from . import abstract_model_test

from sklearn.model_selection import StratifiedKFold

class PytorchModelTest(abstract_model_test.AbstractModelTest):
    def __init__(self, model, model_specs_dict: typing.Dict):
        self._model = model
        self._presaved_models_paths_dict = model_specs_dict.get("presaved_paths")
        self._evaluation_metrics = []

        self._run_id = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_sklearn"

        self._metrics_output_path = f"{model_config_dict['metrics_output_path']}/{self._run_id}"
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

        skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

        # Get item from train data
        X = [item[0] for item in data]
        y = [item[1] for item in data]

        # TODO: Find a better way to do this validation
        # This is to check if y has more than one dimension, for the multiclass case to work with skf.split
        try:
            y = np.array(y).argmax(1)
        except:
            pass

        # TODO: Essa variação por fold também nem faça tanto sentido, uma vez
        # que na verdade ele varia dependendo da quantidade de modelos que foi
        # passado. Então talvez seja melhor fariar por fold index ou algo assim
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print('------------fold no---------{}----------------------'.format(fold))

            # Select the test data
            X_test = X[test_idx]
            y_test = y[test_idx]

            # Load the current fold model
            model_filename = f"{self._presaved_models_paths_dict['paths'][f'{fold}']}"
            self._model = pickle.load(open(model_filename, 'rb'))

            # Get the train metrics
            test_metrics = self.__validate_model(X_test, y_test)
            test_metrics = [fold, *test_metrics]

            # Append the metrics to be further exported
            self._metrics_list.append(test_metrics)

            self._model.reset()

        metrics_df = pd.DataFrame(self._metrics_list, columns=["fold", "acc", "f1", "prec", "recall", "roc_auc", "inference_time", "model_size"])
        metrics_df.to_csv(f"{self._metrics_output_path}/test_metrics_sklearn_{self._model_name}.csv")
