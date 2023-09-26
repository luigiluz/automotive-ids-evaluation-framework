import torch
import random
import typing

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

        self._evaluation_metrics = []

        self._metrics_output_path = model_config_dict['metrics_output_path']
        self._models_output_path = model_config_dict['models_output_path']

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


    def __train_model(self, criterion, device, trainloader, fold):
        self._model.train()

        self._model = self._model.to(device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)

        accuracy_metric = BinaryAccuracy().to(device)

        for epoch in range(self._num_epochs):
            for batch_idx, (data, target) in enumerate(trainloader):
                # get current input and ouputs
                data, target = data.float(), target.reshape(-1, 1).float()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = self._model(data)
                loss = criterion(output, target)
                loss.backward()

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

        # Referencia para salvar os modelos
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        self._model.eval()

        # Descomentar isso aqui caso queira salvar o modelo
        torch.save(self._model.state_dict(), f"{self._models_output_path}/pytorch_model_{self._model_name}_{fold}")


    def __validate_model(self, criterion, device, testloader, fold):
        self._model.eval()
        test_loss = 0

        accuracy_metric = BinaryAccuracy().to(device)
        f1_score_metric = BinaryF1Score().to(device)
        auc_roc_metric = BinaryAUROC().to(device)
        precision_score = BinaryPrecision().to(device)
        recall_score = BinaryRecall().to(device)

        with torch.no_grad():
            for data, target in testloader:
                # traz os dados pro dispositivo
                data, target = data.float(), target.reshape(-1, 1).float()
                output = self._model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss

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
            metrics_df = pd.DataFrame(self._evaluation_metrics, columns=["fold", "acc", "prec", "recall", "f1", "roc_auc"])
            metrics_df.to_csv(f"{self._metrics_output_path}/{self._model_name}_BS{self._batch_size}_EP{self._num_epochs}_LR{self._learning_rate}.csv")

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
        criterion = nn.BCELoss()

        skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

        # Get item from train data
        X = [item[0] for item in train_data]
        y = [item[1] for item in train_data]

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

            self.__train_model(criterion, device, trainloader, fold)

            self.__validate_model(criterion, device, testloader, fold)
