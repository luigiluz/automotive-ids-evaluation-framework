#!/bin/bash

# Paths
BASE_PATH="/home/lfml/workspace/automotive-ids-evaluation-framework"
FEAT_GEN_CONFIG_FOLDER="config_jsons/feat_gen"
MODEL_CONFIG_FOLDER="config_jsons/model"
PRESAVED_MODELS_FOLDER="config_jsons/presaved_files"

# Change this according to the desired configuration
SELECTED_FEAT_GEN_CONFIG="cnnids_tow_config_maidai.json"
# SELECTED_MODEL_CONFIG="cnnids_model_maidai.json"
SELECTED_MODEL_CONFIG="multiclass_cnnids_model_maidai.json"

SELECTED_PRESAVED_MODELS="multiclass_cnn.json"

# DO NOT CHANGE FROM THIS POINT ON
FEAT_GEN_CONFIG_PATH="${BASE_PATH}/${FEAT_GEN_CONFIG_FOLDER}/${SELECTED_FEAT_GEN_CONFIG}"
MODEL_CONFIG_PATH="${BASE_PATH}/${MODEL_CONFIG_FOLDER}/${SELECTED_MODEL_CONFIG}"
PRESAVED_MODELS_PATH="${BASE_PATH}/${PRESAVED_MODELS_FOLDER}/${SELECTED_PRESAVED_MODELS}"

# Run the feature generator step
# venv/bin/python3 execute_feature_generator.py --feat_gen_config ${FEAT_GEN_CONFIG_PATH}

# Run the model training and validation step
# venv/bin/python3 execute_model_train_validation.py --feat_gen_config ${FEAT_GEN_CONFIG_PATH} --model_hyperparams ${MODEL_CONFIG_PATH}

# Run the model test step
venv/bin/python3 execute_model_test.py --feat_gen_config ${FEAT_GEN_CONFIG_PATH} --model_hyperparams ${MODEL_CONFIG_PATH} --presaved ${PRESAVED_MODELS_PATH}
