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

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC
)

class PytorchModelTrainValidation(abstract_model_train_validate.AbstractModelTrainValidate):
    def __init__(self, model, model_config_dict: typing.Dict):
        self._model = model

        self._model_name = model_config_dict['model_name']
        self._learning_rate = model_config_dict['learning_rate']
        self._batch_size = model_config_dict['batch_size']
        self._num_epochs = model_config_dict['num_epochs']
        self._criterion = model_config_dict['criterion']

        self._evaluation_metrics = []
        self._train_validation_losses = []

        self._run_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        self._metrics_output_path = f"{model_config_dict['metrics_output_path']}/{self._run_id}"
        if not os.path.exists(self._metrics_output_path):
            os.makedirs(self._metrics_output_path)
            print("Metrics output directory created successfully")

        self._models_output_path = f"{model_config_dict['models_output_path']}/{self._run_id}"
        if not os.path.exists(self._models_output_path):
            os.makedirs(self._models_output_path)
            print("Models output directory created successfully")

        self._early_stopping_patience = model_config_dict['early_stopping_patience']
        self._best_val_loss = float("inf")
        self._epochs_without_improvement = 0

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

    def __reset_early_stopping_counter(self):
        self._epochs_without_improvement = 0

    def __train_model(self, criterion, device, trainloader, fold, epoch) -> int:
        self._model.train()
        train_loss = 0

        self._model = self._model.to(device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)

        accuracy_metric = BinaryAccuracy().to(device)

        for batch_idx, (data, target) in enumerate(trainloader):
            data = data.float()
            if len(target.shape) == 1:
                target = target.reshape(-1, 1)
            target = target.float()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
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

                output = self._model(data)
                val_loss += criterion(output, target).item()  # sum up batch loss

                ret = self.__check_early_stopping(val_loss, testloader)

            return ret, val_loss


    def __test_model(self, criterion, device, testloader, fold):
        self._model.eval()

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
                auc_roc_metric.update(output.detach(), target)
                precision_score.update(output.detach(), target)
                recall_score.update(output.detach(), target)

            # Calculate metrics
            acc = accuracy_metric.compute().cpu().numpy()
            f1 = f1_score_metric.compute().cpu().numpy()
            roc_auc = auc_roc_metric.compute().cpu().numpy()
            prec = precision_score.compute().cpu().numpy()
            recall = recall_score.compute().cpu().numpy()

            # Append metrics on list
            self._evaluation_metrics.append([fold, acc, prec, recall, f1, roc_auc])


    def execute(self, train_data):
        def collate_gpu(batch):
            x, t = torch.utils.data.dataloader.default_collate(batch)
            return x.to(device="cuda:0"), t.to(device="cuda:0")

        # Reset all seed to ensure reproducibility
        self.__seed_all(0)
        g = torch.Generator()
        g.manual_seed(42)

        # Use gpu to train as preference
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

            self._model.apply(self.__reset_weights)

            for epoch in range(self._num_epochs):
                train_loss = self.__train_model(criterion, device, trainloader, fold, epoch)
                ret, val_loss = self.__validate_model(criterion, device, testloader, fold, epoch)
                if (ret < 0):
                    print(f"Early stopping! Validation loss hasn't improved for {self._early_stopping_patience} epochs")
                    break

                self._train_validation_losses.append([fold, epoch, train_loss, val_loss])

            self.__test_model(criterion, device, testloader, fold)

            # Reset early stopping counter for next fold
            self.__reset_early_stopping_counter()

            # Save model
            self.__save_model_state_dict(fold)

            # Export metrics
            metrics_df = pd.DataFrame(self._evaluation_metrics, columns=["fold", "acc", "prec", "recall", "f1", "roc_auc"])
            metrics_df.to_csv(f"{self._metrics_output_path}/val_metrics_{self._model_name}_BS{self._batch_size}_EP{self._num_epochs}_LR{self._learning_rate}.csv")

            train_val_loss_df = pd.DataFrame(self._train_validation_losses, columns=["fold", "epoch", "train_loss", "val_loss"])
            train_val_loss_df.to_csv(f"{self._metrics_output_path}/train_val_losses_{self._model_name}_BS{self._batch_size}_EP{self._num_epochs}_LR{self._learning_rate}.csv")
