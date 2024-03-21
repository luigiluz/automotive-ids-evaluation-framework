import argparse
import json

from feature_generator import cnn_ids_feature_generator

AVAILABLE_FEATURE_GENERATORS = {
    "CNNIDSFeatureGenerator": cnn_ids_feature_generator.CNNIDSFeatureGenerator
}

def main():
    parser = argparse.ArgumentParser(description='Execute feature generation step')
    parser.add_argument('--feat_gen_config', required=True, help='JSON File containing the configs for the specified feature generation method')
    parser.add_argument('--bench_time', action='store_true', help='Flag to execute the feature generator execution time benchmark')
    args = parser.parse_args()

    try:
        with open(args.feat_gen_config, 'r') as feat_gen_config:
            feat_gen_config_dict = json.load(feat_gen_config)

    except FileNotFoundError as e:
        print(f"parse_args: Error: {e}")
    except json.JSONDecodeError as e:
        print(f"parse_args: Error decoding JSON: {e}")

    print("##### Loaded paths dictionary #####")
    print(json.dumps(feat_gen_config_dict, indent=4, sort_keys=True))

    feature_generator_name = feat_gen_config_dict['feature_generator']
    feature_generator_config = feat_gen_config_dict['config']
    feature_generator_paths = feat_gen_config_dict['paths']

    if feature_generator_name not in AVAILABLE_FEATURE_GENERATORS:
        raise KeyError(f"Selected feature generator: {feature_generator_name} is NOT available!")

    selected_feature_generator = AVAILABLE_FEATURE_GENERATORS[feature_generator_name](feature_generator_config)
    print(f"> Selected feature generator: {feature_generator_name}")

    if args.bench_time:
        print("> Execution time benchmark generation")
        selected_feature_generator.benchmark_execution_time()
    else:
        print("> Generating features...")
        selected_feature_generator.generate_features(feature_generator_paths)

    print("Feature generator successfully executed!")


if __name__ == "__main__":
    main()
