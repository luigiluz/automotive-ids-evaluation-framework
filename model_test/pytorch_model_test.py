import os
import torch
import random
import typing
import datetime

import pandas as pd
import numpy as np

from . import abstract_model_test

from sklearn.model_selection import StratifiedKFold

class PytorchModelTest(abstract_model_test.AbstractModelTest):
    def __init__(self, model, presaved_models_state_dict: typing.Dict):
        self._model = model
        self._presaved_models_state_dict = presaved_models_state_dict.get("presaved_models")
        self._evaluation_metrics = []
        self._batch_size = 64
        self._output_path = presaved_models_state_dict.get("output_path")

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

        np.savez(f"{self._output_path}/fold_{fold}_cnn_output.npz", store_tensor.numpy())

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

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print('------------fold no---------{}----------------------'.format(fold))

            # train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

            # trainloader = torch.utils.data.DataLoader(
            #             data,
            #             batch_size=self._batch_size,
            #             sampler=train_subsampler,
            #             generator=g,
            #             worker_init_fn=self.__seed_worker,
            #             collate_fn=collate_gpu)
            testloader = torch.utils.data.DataLoader(
                        data,
                        batch_size=self._batch_size,
                        sampler=test_subsampler,
                        generator=g,
                        worker_init_fn=self.__seed_worker,
                        collate_fn=collate_gpu)


            self._model.load_state_dict(torch.load(self._presaved_models_state_dict[f"{fold}"], map_location='cpu'))
            self._model.to(device)

            self.__model_cnn_forward(device, testloader, fold)

