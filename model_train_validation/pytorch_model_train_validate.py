import os
import torch
import random
import typing
import datetime

import pandas as pd
import numpy as np

from torch import nn

from sklearn.model_selection import StratifiedKFold

from . import abstract_model_train_validate
from custom_metrics import timing, storage

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

class PytorchModelTrainValidation(abstract_model_train_validate.AbstractModelTrainValidate):
    def __init__(self, model, model_config_dict: typing.Dict):
        self._model = model

        self._model_name = model_config_dict['model_name']
        self._criterion = model_config_dict['criterion']
        self._model_specs_dict = model_config_dict

        hyperparameters_dict = model_config_dict.get('hyperparameters')
        self._learning_rate = hyperparameters_dict['learning_rate']
        self._batch_size = hyperparameters_dict['batch_size']
        self._num_epochs = hyperparameters_dict['num_epochs']

        self._evaluation_metrics = []
        self._train_validation_losses = []

        self._run_id = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_pytorch_train"
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

        self._early_stopping_patience = hyperparameters_dict['early_stopping_patience']
        self._best_val_loss = float("inf")
        self._epochs_without_improvement = 0

        self._number_of_outputs = hyperparameters_dict.get('num_outputs', -1)

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


    def __reset_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()


    def __save_model_state_dict(self, fold):
        self._model.eval()
        torch.save(self._model.state_dict(), f"{self._models_output_path}/pytorch_model_{self._model_name}_{fold}")

    def __check_early_stopping(self, val_loss, testloader) -> int:
        ret = 0
        # Early stopping update
        val_loss = val_loss / len(testloader)
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement = self._epochs_without_improvement + 1

        # Early stopping condition
        if self._epochs_without_improvement >= self._early_stopping_patience:
            ret = -1

        return ret

    def __reset_early_stopping(self):
        self._best_val_loss = float("inf")
        self._epochs_without_improvement = 0

    def __train_model(self, criterion, device, trainloader, fold, epoch) -> int:
        self._model.train()
        train_loss = 0

        self._model = self._model.to(device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)

        if self._number_of_outputs > 1:
            accuracy_metric = MulticlassAccuracy(num_classes=self._number_of_outputs).to(device)
        else:
            accuracy_metric = BinaryAccuracy().to(device)

        for batch_idx, (data, target) in enumerate(trainloader):
            data = data.float()
            if len(target.shape) == 1:
                target = target.reshape(-1, 1)
            target = target.float()
            # zero the parameter gradients
            optimizer.zero_grad()

            # # forward + backward + optimize
            # # TODO: Later think a way this to be included inside model
            # if (self._model_name == "MultiStageIDS"):
            #     # Run stages
            #     y1 = self._model.forward_first_stage(data)
            #     y2 = self._model.forward_second_stage(data)

            #     # Move to devices
            #     y1 = y1.to(device)
            #     y2 = y2.to(device)

            #     print(f"y1 = {y1}")
            #     print(f"y1.shape = {y1.shape}")
            #     print(f"y2 = {y2}")
            #     print(f"y2.shape = {y2.shape}")

            #     # Combine data
            #     data = torch.cat((y1, y2), axis=1)

            output = self._model(data)
            loss = criterion(output, target)
            loss.backward()
            train_loss += loss.item()

            ## update model params
            optimizer.step()

            output = output.detach().round()
            acc = accuracy_metric(output, target)

            ## metrics logs
            if batch_idx % 1000 == 0:
                # accuracy = 100 * correct / len(trainloader)
                print('Train Fold: {} \t Epoch: {} \t[{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc: {:.6f}'.format(
                fold,epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item(), acc))

        train_loss = train_loss / len(trainloader)

        return train_loss


    def __validate_model(self, criterion, device, testloader, fold, epoch) -> typing.Tuple[int, float]:
        self._model.eval()
        ret = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in testloader:
                data = data.float()
                if len(target.shape) == 1:
                    target = target.reshape(-1, 1)
                target = target.float()

                # if (self._model_name == "MultiStageIDS"):
                #     # Run stages
                #     y1 = self._model.forward_first_stage(data)
                #     y2 = self._model.forward_second_stage(data)

                #     # Move to devices
                #     y1 = y1.to(device)
                #     y2 = y2.to(device)

                #     # Combine data
                #     data = torch.cat((y1, y2), axis=1)

                output = self._model(data)
                val_loss += criterion(output, target).item()  # sum up batch loss

        ret = self.__check_early_stopping(val_loss, testloader)

        return ret, val_loss


    def __test_model(self, criterion, device, testloader, fold):
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

                # if (self._model_name == "MultiStageIDS"):
                #     # Run stages
                #     y1 = self._model.forward_first_stage(data)
                #     y2 = self._model.forward_second_stage(data)

                #     # Move to devices
                #     y1 = y1.to(device)
                #     y2 = y2.to(device)

                #     # Combine data
                #     data = torch.cat((y1, y2), axis=1)

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


    def execute(self, train_data):
        def collate_gpu(batch):
            x, t = torch.utils.data.dataloader.default_collate(batch)
            return x.to(device="cuda:1"), t.to(device="cuda:1")
            # return x.to(device="cpu"), t.to(device="cpu")

        # Reset all seed to ensure reproducibility
        self.__seed_all(0)
        g = torch.Generator()
        g.manual_seed(42)

        # Use gpu to train as preference
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")

        # Get this criterion from configuration parameter
        criterion = None
        if (self._criterion == 'binary-cross-entropy'):
            criterion = nn.BCELoss()
        elif (self._criterion == 'categorical-cross-entropy'):
            criterion = nn.CrossEntropyLoss()
        else:
            raise KeyError(f"Selected criterion : {self._criterion} is NOT available!")

        skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

        # Get item from train data
        X = [item[0] for item in train_data]
        y = [item[1] for item in train_data]

        # TODO: Find a better way to do this validation
        # This is to check if y has more than one dimension, for the multiclass case to work with skf.split
        try:
            y = np.array(y).argmax(1)
        except:
            pass

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print('------------fold no---------{}----------------------'.format(fold))

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

            trainloader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size=self._batch_size,
                        sampler=train_subsampler,
                        generator=g,
                        worker_init_fn=self.__seed_worker,
                        collate_fn=collate_gpu)
            testloader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size=self._batch_size,
                        sampler=test_subsampler,
                        generator=g,
                        worker_init_fn=self.__seed_worker,
                        collate_fn=collate_gpu)

            # TODO: adicionar o carregamento dos modelos
            self._model.apply(self.__reset_weights)
            if (self._model_name == "MultiStageIDS"):
                random_forest_path = self._model_specs_dict["first_stage"]["presaved_paths"][f"{fold}"]
                pruned_cnn_path = self._model_specs_dict["second_stage"]["presaved_paths"][f"{fold}"]
                self._model.load_stages_models(random_forest_path, pruned_cnn_path)

            for epoch in range(self._num_epochs):
                train_loss = self.__train_model(criterion, device, trainloader, fold, epoch)
                ret, val_loss = self.__validate_model(criterion, device, testloader, fold, epoch)
                if (ret < 0):
                    print(f"Early stopping! Validation loss hasn't improved for {self._early_stopping_patience} epochs")
                    break

                self._train_validation_losses.append([fold, epoch, train_loss, val_loss])

            self.__test_model(criterion, device, testloader, fold)

            # Reset early stopping for next fold
            self.__reset_early_stopping()

            # Save model
            self.__save_model_state_dict(fold)

            # Export metrics
            metrics_df = pd.DataFrame(self._evaluation_metrics, columns=["fold", "acc", "prec", "recall", "f1", "roc_auc", "inference_time", "model_size"])
            metrics_df.to_csv(f"{self._metrics_output_path}/val_metrics_{self._model_name}_BS{self._batch_size}_EP{self._num_epochs}_LR{self._learning_rate}.csv")

            train_val_loss_df = pd.DataFrame(self._train_validation_losses, columns=["fold", "epoch", "train_loss", "val_loss"])
            train_val_loss_df.to_csv(f"{self._metrics_output_path}/train_val_losses_{self._model_name}_BS{self._batch_size}_EP{self._num_epochs}_LR{self._learning_rate}.csv")
