import argparse
import json
import torch

from feature_generator import cnn_ids_feature_generator
from models import conv_net_ids, multiclass_conv_net_ids
from model_test import pytorch_model_test

AVAILABLE_FEATURE_GENERATORS = {
    "CNNIDSFeatureGenerator": cnn_ids_feature_generator.CNNIDSFeatureGenerator
}

AVAILABLE_IDS = {
    "CNNIDS": conv_net_ids.ConvNetIDS,
    "MultiClassCNNIDS": multiclass_conv_net_ids.MultiClassConvNetIDS
}

AVAILABLE_FRAMEWORKS = {
    "pytorch": pytorch_model_test.PytorchModelTest
}

def main():
    print("Executing main function...")
    parser = argparse.ArgumentParser(description='Execute model train validation step')
    parser.add_argument('--feat_gen_config', required=True, help='JSON File containing the configs for the specified feature generation method')
    parser.add_argument('--model_hyperparams', required=True, help='JSON File containing the model hyperparams')
    parser.add_argument('--presaved', required=True, help="JSON File containing the path for the presaved models")
    args = parser.parse_args()

    try:
        with open(args.feat_gen_config, 'r') as feat_gen_config:
            feat_gen_config_dict = json.load(feat_gen_config)

        with open(args.model_hyperparams, 'r') as model_hyperparams:
            model_hyperparams_dict = json.load(model_hyperparams)

        with open(args.presaved, 'r') as presaved_models:
            presaved_models_dict = json.load(presaved_models)

    except FileNotFoundError as e:
        print(f"parse_args: Error: {e}")
    except json.JSONDecodeError as e:
        print(f"parse_args: Error decoding JSON: {e}")

    print("##### Loaded configuration files #####")
    print(json.dumps(feat_gen_config_dict, indent=4, sort_keys=True))
    print("###############")
    print(json.dumps(model_hyperparams_dict, indent=4, sort_keys=True))
    print("###############")
    print(json.dumps(presaved_models_dict, indent=4, sort_keys=True))
    print("###############")

    print("> Loading features...")
    feature_generator_name = feat_gen_config_dict['feature_generator']
    feature_generator_config = feat_gen_config_dict['config']
    feature_generator_load_paths = feat_gen_config_dict['load_paths']

    if feature_generator_name not in AVAILABLE_FEATURE_GENERATORS:
        raise KeyError(f"Selected feature generator: {feature_generator_name} is NOT available!")

    selected_feature_generator = AVAILABLE_FEATURE_GENERATORS[feature_generator_name](feature_generator_config)
    data = selected_feature_generator.load_features(feature_generator_load_paths)

    print("> Creating model...")
    model_name = model_hyperparams_dict['model']
    if model_name not in AVAILABLE_IDS:
        raise KeyError(f"Selected model: {model_name} is NOT available!")

    num_outputs = model_hyperparams_dict.get('num_outputs', 1)
    if num_outputs > 1:
        model = AVAILABLE_IDS[model_name](number_of_outputs=num_outputs)
    else:
        model = AVAILABLE_IDS[model_name]()

    print(f"> Empty {model_name} was created with {num_outputs} outputs")

    print(f"> Loading pre-saved model from ... folder")
    # presaved_model_path = "/home/lfml/workspace/models/2023_10_02_22_14_33/pytorch_model_MultiClassCNNIDS_0"
    # model = model.load_state_dict(torch.load(presaved_model_path))

    print(f"Model = {model}")

    print("> Initializing model test...")

    framework = model_hyperparams_dict['framework']
    if framework not in AVAILABLE_FRAMEWORKS:
        raise KeyError(f"Selected framework: {framework} is NOT available!")

    test = AVAILABLE_FRAMEWORKS["pytorch"](model, presaved_models_dict)
    test.execute(data)

    print("Model tested successfully!")

if __name__ == "__main__":
    main()
