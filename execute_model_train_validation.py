import argparse
import json

from feature_generator import cnn_ids_feature_generator
from models import conv_net_ids, multiclass_conv_net_ids
from model_train_validation import pytorch_model_train_validate

AVAILABLE_FEATURE_GENERATORS = {
    "CNNIDSFeatureGenerator": cnn_ids_feature_generator.CNNIDSFeatureGenerator
}

AVAILABLE_IDS = {
    "CNNIDS": conv_net_ids.ConvNetIDS,
    "MultiClassCNNIDS": multiclass_conv_net_ids.MultiClassConvNetIDS
}

# TODO: adicionar classe que pega os itens do pytorch
AVAILABLE_FRAMEWORKS = {
    "pytorch": pytorch_model_train_validate.PytorchModelTrainValidation
}

def main():
    parser = argparse.ArgumentParser(description='Execute model train validation step')
    parser.add_argument('--feat_gen_config', required=True, help='JSON File containing the configs for the specified feature generation method')
    parser.add_argument('--model_hyperparams', required=True, help='JSON File containing the model hyperparams')
    args = parser.parse_args()

    try:
        with open(args.feat_gen_config, 'r') as feat_gen_config:
            feat_gen_config_dict = json.load(feat_gen_config)

        with open(args.model_hyperparams, 'r') as model_hyperparams:
            model_hyperparams_dict = json.load(model_hyperparams)

    except FileNotFoundError as e:
        print(f"parse_args: Error: {e}")
    except json.JSONDecodeError as e:
        print(f"parse_args: Error decoding JSON: {e}")

    print("##### Loaded configuration files #####")
    print(json.dumps(feat_gen_config_dict, indent=4, sort_keys=True))
    print("###############")
    print(json.dumps(model_hyperparams_dict, indent=4, sort_keys=True))
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

    print(f"> {model_name} was created with {num_outputs} outputs")

    print("> Initializing model training and evaluation...")

    framework = model_hyperparams_dict['framework']
    if framework not in AVAILABLE_FRAMEWORKS:
        raise KeyError(f"Selected framework: {framework} is NOT available!")

    train_validator = AVAILABLE_FRAMEWORKS["pytorch"](model, model_hyperparams_dict)
    train_validator.execute(data)

    print("Model trained successfully!")


if __name__ == "__main__":
    main()
