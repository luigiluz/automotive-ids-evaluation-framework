#!/bin/bash

# Paths
BASE_PATH="/home/lfml/workspace/automotive-ids-evaluation-framework"
FEAT_GEN_CONFIG_FOLDER="config_jsons/feat_gen"
MODEL_CONFIG_FOLDER="config_jsons/model"
PRESAVED_MODELS_FOLDER="config_jsons/presaved_files"

# Change this according to the desired configuration
## Feat generator configs
SELECTED_FEAT_GEN_CONFIG="cnnids_config_maidai.json"

## Model configs
CNN_PRESAVED_CONFIG="ms_cnn_presaved.json"
RF_PRESAVED_CONFIG="ms_rf_presaved.json"

# DO NOT CHANGE FROM THIS POINT ON
FEAT_GEN_CONFIG_PATH="${BASE_PATH}/${FEAT_GEN_CONFIG_FOLDER}/${SELECTED_FEAT_GEN_CONFIG}"
CNN_MODELS_PATH="${BASE_PATH}/${PRESAVED_MODELS_FOLDER}/${CNN_PRESAVED_CONFIG}"
RF_MODELS_PATH="${BASE_PATH}/${PRESAVED_MODELS_FOLDER}/${RF_PRESAVED_CONFIG}"

# Run the multi stage deep learning model inference
venv/bin/python3 execute_multi_stage_ids.py --feat_gen_config ${FEAT_GEN_CONFIG_PATH} --cnn_presaved_paths ${CNN_MODELS_PATH} --rf_presaved_paths ${RF_MODELS_PATH}
