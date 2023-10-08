import argparse
import json
import pickle
import os
import torch
import random
import datetime

import numpy as np
import pandas as pd

from feature_generator import cnn_ids_feature_generator
from models import (
    conv_net_ids,
    multiclass_conv_net_ids,
    pruned_conv_net_ids,
    sklearn_classifier,
    multi_stage_ids
)
from model_train_validation import (
    pytorch_model_train_validate,
    sklearn_model_train_validate
)

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

from custom_metrics import general

AVAILABLE_FEATURE_GENERATORS = {
    "CNNIDSFeatureGenerator": cnn_ids_feature_generator.CNNIDSFeatureGenerator
}

AVAILABLE_IDS = {
    "CNNIDS": conv_net_ids.ConvNetIDS,
    "MultiClassCNNIDS": multiclass_conv_net_ids.MultiClassConvNetIDS,
    "PrunedCNNIDS": pruned_conv_net_ids.PrunedConvNetIDS,
    "SklearnClassifier": sklearn_classifier.SklearnClassifier
}

AVAILABLE_FRAMEWORKS = {
    "pytorch": pytorch_model_train_validate.PytorchModelTrainValidation,
    "sklearn": sklearn_model_train_validate.SklearnModelTrainValidation
}

# Helper functions
def __seed_all(seed):
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


def __seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_gpu(batch):
    x, t = torch.utils.data.dataloader.default_collate(batch)
    return x.to(device="cuda:0"), t.to(device="cuda:0")
    # return x.to(device="cpu"), t.to(device="cpu")

# Main functions
def main():
    parser = argparse.ArgumentParser(description='Execute multi stage deep learning based IDS')

    parser.add_argument('--multi_stage_ids_config', required=True, help='JSON File containing the config for the multi stage ids process')
    args = parser.parse_args()

    try:
        with open(args.multi_stage_ids_config, 'r') as multi_stage_ids_config:
            multi_stage_ids_config_dict = json.load(multi_stage_ids_config)

    except FileNotFoundError as e:
        print(f"parse_args: Error: {e}")
    except json.JSONDecodeError as e:
        print(f"parse_args: Error decoding JSON: {e}")

    print("##### Loaded configuration files #####")
    print(json.dumps(multi_stage_ids_config_dict, indent=4, sort_keys=True))

    # Create folder
    run_id = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_multistage"

    metrics_output_path = f"/home/lfml/workspace/metrics/{run_id}"
    if not os.path.exists(metrics_output_path):
        os.makedirs(metrics_output_path)
        print("Metrics output directory created successfully")

    print(">Executing multi-stage deep learning based IDS")

    feat_gen_config_dict = multi_stage_ids_config_dict['feat_gen']
    first_stage_config_dict = multi_stage_ids_config_dict['first_stage']
    second_stage_config_dict = multi_stage_ids_config_dict['second_stage']

    feature_generator_name = feat_gen_config_dict['feature_generator']
    feature_generator_config = feat_gen_config_dict['config']
    feature_generator_load_paths = feat_gen_config_dict['load_paths']

    if feature_generator_name not in AVAILABLE_FEATURE_GENERATORS:
        raise KeyError(f"Selected feature generator: {feature_generator_name} is NOT available!")

    print("> Loading features...")
    selected_feature_generator = AVAILABLE_FEATURE_GENERATORS[feature_generator_name](feature_generator_config)
    data = selected_feature_generator.load_features(feature_generator_load_paths)
    cnn_model_name = first_stage_config_dict['model_name']
    if cnn_model_name not in AVAILABLE_IDS:
        raise KeyError(f"Selected model: {cnn_model_name} is NOT available!")

    # Get item from train data
    X = [item[0] for item in data]
    y = [item[1] for item in data]

    # TODO: Find a better way to do this validation
    # This is to check if y has more than one dimension, for the multiclass case to work with skf.split
    try:
        y = np.array(y).argmax(1)
    except:
        pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    BATCH_SIZE = first_stage_config_dict['hyperparameters']['batch_size']
    evaluation_metrics = []

    # Reset all seed to ensure reproducibility
    __seed_all(0)
    g = torch.Generator()
    g.manual_seed(42)

    for fold_index in first_stage_config_dict['presaved_paths'].keys():
        print('------------fold no---------{}----------------------'.format(fold_index))
        # Load presaved models
        ## Load first stage model
        cnn_model = AVAILABLE_IDS[cnn_model_name]()
        cnn_model.load_state_dict(torch.load(first_stage_config_dict['presaved_paths'][fold_index], map_location='cpu'))
        cnn_model.to(device)

        ## Load second stage model
        random_forest_model = pickle.load(open(f"{second_stage_config_dict['presaved_paths'][fold_index]}", 'rb'))

        # Create multi stage ids based on presaved models
        ms_ids_model = multi_stage_ids.MultiStageIDS(cnn_model, random_forest_model, device)

        testloader = torch.utils.data.DataLoader(
                    data,
                    batch_size=BATCH_SIZE,
                    generator=g,
                    worker_init_fn=__seed_worker,
                    collate_fn=collate_gpu)

        # TODO: Atualizar pros casos de multiplas saídas do modelo de Deep Learning
        # TODO: Adicionar calculo de tempo de inferência
        metrics_computer = {
            "rf": general.GeneralMetrics("rf", "pytorch", 1, device),
            "dl": general.GeneralMetrics("dl", "pytorch", 1, device),
            "ms": general.GeneralMetrics("ms", "pytorch", 1, device)
        }

        with torch.no_grad():
            for data, target in testloader:
                data = data.float()
                if len(target.shape) == 1:
                    target = target.reshape(-1, 1)
                target = target.float()

                output = ms_ids_model.forward(data)
                for key in output.keys():
                    output[key].to(device)

                for metrics_index in metrics_computer.keys():
                    metrics_computer[metrics_index].update(output[metrics_index], target)

        for metrics_index in metrics_computer.keys():
            metrics_computer[metrics_index].compute()
            metrics_list = [fold_index, *metrics_computer[metrics_index].get_as_list()]
            evaluation_metrics.append(metrics_list)

        metrics_df = pd.DataFrame(evaluation_metrics, columns=["fold", "step", "f1", "acc", "prec", "recall", "roc_auc", "inference time", "storage size"])
        metrics_df.to_csv(f"{metrics_output_path}/multistage_test.csv")


if __name__ == "__main__":
    main()
