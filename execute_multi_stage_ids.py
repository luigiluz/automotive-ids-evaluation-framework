import argparse
import json
import pickle
import torch
import random

import numpy as np

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

from sklearn.model_selection import StratifiedKFold

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
    # return x.to(device="cuda:0"), t.to(device="cuda:0")
    return x.to(device="cpu"), t.to(device="cpu")

# Main functions
def main():
    parser = argparse.ArgumentParser(description='Execute multi stage deep learning based IDS')

    parser.add_argument('--feat_gen_config', required=True, help='JSON File containing the configs for the specified feature generation method')
    parser.add_argument('--cnn_presaved_paths', required=True, help='JSON File containing the path for the presaved cnn models')
    parser.add_argument('--rf_presaved_paths', required=True, help='JSON File containing the path for the prevsaved rf models')
    args = parser.parse_args()

    try:
        with open(args.feat_gen_config, 'r') as feat_gen_config:
            feat_gen_config_dict = json.load(feat_gen_config)

        with open(args.cnn_presaved_paths, 'r') as cnn_presaved:
            cnn_presaved_dict = json.load(cnn_presaved)

        with open(args.rf_presaved_paths, 'r') as rf_presaved:
            rf_presaved_dict = json.load(rf_presaved)

    except FileNotFoundError as e:
        print(f"parse_args: Error: {e}")
    except json.JSONDecodeError as e:
        print(f"parse_args: Error decoding JSON: {e}")

    print("##### Loaded configuration files #####")
    print(json.dumps(feat_gen_config_dict, indent=4, sort_keys=True))
    print("###############")
    print(json.dumps(cnn_presaved_dict, indent=4, sort_keys=True))
    print("###############")
    print(json.dumps(rf_presaved_dict, indent=4, sort_keys=True))
    print("###############")

    print(">Executing multi-stage deep learning based IDS")

    feature_generator_name = feat_gen_config_dict['feature_generator']
    feature_generator_config = feat_gen_config_dict['config']
    feature_generator_load_paths = feat_gen_config_dict['load_paths']

    if feature_generator_name not in AVAILABLE_FEATURE_GENERATORS:
        raise KeyError(f"Selected feature generator: {feature_generator_name} is NOT available!")

    print("> Loading features...")
    selected_feature_generator = AVAILABLE_FEATURE_GENERATORS[feature_generator_name](feature_generator_config)
    data = selected_feature_generator.load_features(feature_generator_load_paths)
    cnn_model_name = cnn_presaved_dict['model_name']
    if cnn_model_name not in AVAILABLE_IDS:
        raise KeyError(f"Selected model: {cnn_model_name} is NOT available!")

    print("> Executing validation step...")
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

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    BATCH_SIZE = 64

    # Reset all seed to ensure reproducibility
    __seed_all(0)
    g = torch.Generator()
    g.manual_seed(42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # Load presaved models
        random_forest_model = pickle.load(open(f"{rf_presaved_dict['paths'][f'{fold}']}", 'rb'))
        cnn_model = AVAILABLE_IDS[cnn_model_name]()
        cnn_model.load_state_dict(torch.load(cnn_presaved_dict['paths'][f"{fold}"], map_location='cpu'))
        cnn_model.to(device)

        # Create multi stage ids based on presaved models
        ms_ids_model = multi_stage_ids.MultiStageIDS(cnn_model, random_forest_model)

        # TODO: Carregar os modelos de acordo com os que estão nos arquivos

        print('------------fold no---------{}----------------------'.format(fold))

        # train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        # trainloader = torch.utils.data.DataLoader(
        #             data,
        #             batch_size=BATCH_SIZE,
        #             sampler=train_subsampler,
        #             generator=g,
        #             worker_init_fn=__seed_worker,
        #             collate_fn=collate_gpu)
        testloader = torch.utils.data.DataLoader(
                    data,
                    batch_size=BATCH_SIZE,
                    sampler=test_subsampler,
                    generator=g,
                    worker_init_fn=__seed_worker,
                    collate_fn=collate_gpu)

        store_tensor = torch.Tensor([])

        with torch.no_grad():
            for data, target in testloader:
                data = data.float()
                if len(target.shape) == 1:
                    target = target.reshape(-1, 1)
                target = target.float()

                output = ms_ids_model.forward(data)
                print(f"output = {output}")

                store_tensor = torch.cat((store_tensor, output.cpu()), 0).cpu()
                print(f"store_tensor = {store_tensor}")
                break

        break


if __name__ == "__main__":
    main()
