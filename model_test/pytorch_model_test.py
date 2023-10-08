import os
import torch
import random
import typing
import datetime

import pandas as pd
import numpy as np

from . import abstract_model_test

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
)

class PytorchModelTest(abstract_model_test.AbstractModelTest):
    def __init__(self, model, model_specs_dict: typing.Dict):
        self._model = model
        self._presaved_models_state_dict = model_specs_dict['presaved_paths']
        self._evaluation_metrics = []
        self._batch_size = model_specs_dict['model_specs']['hyperparameters']['batch_size']
        self._number_of_outputs = model_specs_dict['model_specs']['hyperparameters']['num_outputs']
        self._forward_output_path = model_specs_dict['model_specs']['paths']['forward_output_path']

        self._run_id = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_pytorch"

        self._metrics_output_path = f"{model_specs_dict['model_specs']['paths']['metrics_output_path']}/{self._run_id}"
        if not os.path.exists(self._metrics_output_path):
            os.makedirs(self._metrics_output_path)
            print("Metrics output directory created successfully")


    def __seed_all(self, seed):
        # Reference
        # https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
        if not seed:
            seed = 10

        print("[ Using Seed : ", seed, " ]")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def __seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def __model_cnn_forward(self, device, testloader, fold):
        self._model.eval()

        store_tensor = torch.Tensor([]).cpu()

        with torch.no_grad():
            for data, target in testloader:
                data = data.float()
                if len(target.shape) == 1:
                    target = target.reshape(-1, 1)
                target = target.float()

                output = self._model.cnn_forward(data)

                store_tensor = torch.cat((store_tensor, output.cpu()), 0).cpu()

        np.savez(f"{self._forward_output_path}/fold_{fold}_cnn_output.npz", store_tensor.numpy())


    def __test_model(self, device, testloader, fold):
        self._model.eval()

        if self._number_of_outputs > 1:
            accuracy_metric = MulticlassAccuracy(num_classes=self._number_of_outputs).to(device)
            f1_score_metric = MulticlassF1Score(num_classes=self._number_of_outputs).to(device)
            auc_roc_metric = MulticlassAUROC(num_classes=self._number_of_outputs).to(device)
            precision_score = MulticlassPrecision(num_classes=self._number_of_outputs).to(device)
            recall_score = MulticlassRecall(num_classes=self._number_of_outputs).to(device)
        else:
            accuracy_metric = BinaryAccuracy().to(device)
            f1_score_metric = BinaryF1Score().to(device)
            auc_roc_metric = BinaryAUROC().to(device)
            precision_score = BinaryPrecision().to(device)
            recall_score = BinaryRecall().to(device)

        with torch.no_grad():
            for data, target in testloader:
                data = data.float()
                if len(target.shape) == 1:
                    target = target.reshape(-1, 1)
                target = target.float()

                output = self._model(data)

                accuracy_metric.update(output.detach(), target)
                f1_score_metric.update(output.detach(), target)
                # TODO: Find a better way to perform this computation
                if self._number_of_outputs == 6:
                    auc_roc_metric.update(output.detach(), torch.argmax(target, dim=1))
                else:
                    auc_roc_metric.update(output.detach(), target)
                precision_score.update(output.detach(), target)
                recall_score.update(output.detach(), target)

            # Calculate metrics
            acc = accuracy_metric.compute().cpu().numpy()
            f1 = f1_score_metric.compute().cpu().numpy()
            roc_auc = auc_roc_metric.compute().cpu().numpy()
            prec = precision_score.compute().cpu().numpy()
            recall = recall_score.compute().cpu().numpy()

            # TODO: Get the data format using the data
            dummy_input = torch.randn(1, 1, 44, 116, dtype=torch.float).to(device)
            if device.type == "cpu":
                timing_func = timing.pytorch_inference_time_cpu
            else:
                timing_func = timing.pytorch_inference_time_gpu
            inference_time = timing_func(self._model, dummy_input)

            # TODO: Change this to be only used in case model is random forest
            model_size = storage.pytorch_compute_model_size_mb(self._model)

            # Append metrics on list
            self._evaluation_metrics.append([fold, acc, prec, recall, f1, roc_auc, inference_time, model_size])


    def execute(self, data):
        def collate_gpu(batch):
            x, t = torch.utils.data.dataloader.default_collate(batch)
            return x.to(device="cuda:0"), t.to(device="cuda:0")
        # Reset all seed to ensure reproducibility
        self.__seed_all(0)
        g = torch.Generator()
        g.manual_seed(42)

        # Use gpu to train as preference
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for fold_index in self._presaved_models_state_dict.keys():
            print('------------fold no---------{}----------------------'.format(fold_index))

            testloader = torch.utils.data.DataLoader(
                        data,
                        batch_size=self._batch_size,
                        generator=g,
                        worker_init_fn=self.__seed_worker,
                        collate_fn=collate_gpu)

            self._model.load_state_dict(torch.load(self._presaved_models_state_dict[fold_index], map_location='cpu'))
            self._model.to(device)

            # This is only used in case you want to generate data for random forest models
            #self.__model_cnn_forward(device, testloader, fold)

            # Perform test step
            self.__test_model(device, testloader, fold_index)

            # Export metrics
            metrics_df = pd.DataFrame(self._evaluation_metrics, columns=["fold", "acc", "prec", "recall", "f1", "roc_auc", "inference_time", "model_size"])
            metrics_df.to_csv(f"{self._metrics_output_path}/test_metrics_{self._model_name}_BS{self._batch_size}_EP{self._num_epochs}_LR{self._learning_rate}.csv")
