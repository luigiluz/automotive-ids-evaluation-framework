#!/bin/bash

# Paths
BASE_PATH="/home/lfml/workspace/automotive-ids-evaluation-framework"
FEAT_GEN_CONFIG_FOLDER="config_jsons/feat_generator"
MODEL_TRAIN_VALID_CONFIG_FOLDER="config_jsons/model_train_validate"
MODEL_TEST_CONFIG_FOLDER="config_jsons/model_test"
DETECTION_TIME_CONFIG_FOLDER="config_jsons/test_detection_time"

# Change this according to the desired configuration
## Feat generator configs
SELECTED_FEAT_GEN_CONFIG="AVTP_CNNIDS_train.json"
# SELECTED_FEAT_GEN_CONFIG="TOW_CNNIDS_Multiclass_train.json"
# SELECTED_FEAT_GEN_CONFIG="TOW_CNNIDS_Oneclass_train.json"

## Model train validate configs
# SELECTED_MODEL_TRAIN_VALIDATE_CONFIG="AVTP_CNNIDS_train.json"
# SELECTED_MODEL_TRAIN_VALIDATE_CONFIG="AVTP_PrunedCNNIDS_train.json"
# SELECTED_MODEL_TRAIN_VALIDATE_CONFIG="TOW_PrunedCNNIDS_Multiclass_train.json"
# SELECTED_MODEL_TRAIN_VALIDATE_CONFIG="AVTP_RandomForest_train.json"
# SELECTED_MODEL_TRAIN_VALIDATE_CONFIG="TOW_RandomForest_train.json"

## Model test config
# SELECTED_MODEL_TEST_CONFIG="AVTP_RandomForest_test.json"
# SELECTED_MODEL_TEST_CONFIG="AVTP_PrunedCNNIDS_test.json"

# SELECTED_MODEL_TEST_CONFIG="TOW_MC_PrunedCNNIDS_test.json"
# SELECTED_MODEL_TEST_CONFIG="TOW_RandomForest_test.json"

# SELECTED_MODEL_TEST_CONFIG="TOW_MC_PrunedCNNIDS_test_multiple_folds.json"

## Detection time IDS
SELECTED_DETECTION_TIME_IDS_CONFIG="AVTP_PrunedCNNIDS_detection_time.json"
SELECTED_DETECTION_TIME_IDS_CONFIG="AVTP_RandomForest_detection_time.json"

# DO NOT CHANGE FROM THIS POINT ON
FEAT_GEN_CONFIG_PATH="${BASE_PATH}/${FEAT_GEN_CONFIG_FOLDER}/${SELECTED_FEAT_GEN_CONFIG}"
MODEL_TRAIN_VALID_CONFIG_PATH="${BASE_PATH}/${MODEL_TRAIN_VALID_CONFIG_FOLDER}/${SELECTED_MODEL_TRAIN_VALIDATE_CONFIG}"
MODEL_TEST_CONFIG_PATH="${BASE_PATH}/${MODEL_TEST_CONFIG_FOLDER}/${SELECTED_MODEL_TEST_CONFIG}"
DETECTION_TIME_CONFIG_PATH="${BASE_PATH}/${DETECTION_TIME_CONFIG_FOLDER}/${SELECTED_DETECTION_TIME_IDS_CONFIG}"

# Run the feature generator step
venv/bin/python3 execute_feature_generator.py --feat_gen_config ${FEAT_GEN_CONFIG_PATH} --bench_time

# Run the model training and validation step
# venv/bin/python3 execute_model_train_validation.py --model_train_valid_config ${MODEL_TRAIN_VALID_CONFIG_PATH}

# Run the model test step
# venv/bin/python3 execute_model_test.py --model_test_config ${MODEL_TEST_CONFIG_PATH}

# Run the multi stage method
# venv/bin/python3 execute_multi_stage_ids.py --multi_stage_ids_config ${MULTI_STAGE_CONFIG_PATH}

# Run the detection time experiments
# venv/bin/python3 execute_model_test.py --model_test_config ${DETECTION_TIME_CONFIG_PATH}
