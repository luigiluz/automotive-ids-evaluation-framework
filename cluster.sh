#!/bin/bash
#SBATCH --job-name=ids-run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p short-simple
#SBATCH --qos=simple
#SBATCH --signal=USR1@5
set -e

cdir=${TMPDIR}/detec${SLURM_JOB_ID}
mkdir -p "$cdir" 
chmod 700 "$cdir"
cd "$cdir"

function clean_up {
    echo ">> Iniciando sincronização de arquivos e limpeza do node..."
    # before cleanup rsync data back to home directory
    sync_src="$cdir/automotive-ids-evaluation-framework-deteccao_intrusao"
    if [ -d "$sync_src" ]; then
        rsync -av --exclude='*__pycache__*' --exclude='slurm*' --exclude='.git' "$sync_src/" ~/automotive-ids-evaluation-framework-deteccao_intrusao/
    else
        echo "Warning: source directory '$sync_src' does not exist, skipping rsync."
    fi

   # Apenas sobe para a pasta pai para poder deletar a temporária com segurança
    cd ..
    rm -rf "$cdir"
    echo ">> Limpeza concluída."
}

trap 'clean_up' EXIT SIGTERM

module purge
module use /opt/easybuild/modules/all
module load flex Bison zlib OpenSSL/1.1
module load Python/3.10.8-GCCcore-12.2.0 Xvfb freeglut glew

if [ ! -d ~/automotive-ids-evaluation-framework-deteccao_intrusao ]; then
    echo "Missing automotive-ids-evaluation-framework-deteccao_intrusao directory. Exiting."
    exit 1
fi

rsync -av --exclude='.nfs*' ~/automotive-ids-evaluation-framework-deteccao_intrusao .
cd automotive-ids-evaluation-framework-deteccao_intrusao

if [ ! -d ~/ids_venv ]; then
    python3 -m venv ~/ids_venv
    source ~/ids_venv/bin/activate
    python -m pip install -r requirements.txt
    python -m pip install gdown # Install gdown to download files from Google Drive
    deactivate
    echo "Virtual environment created and dependencies installed."
else
    echo "Virtual environment already exists. Skipping creation."
fi

# download datasets into dataset directory if not already present
if [ ! -d ~/automotive-ids-evaluation-framework-deteccao_intrusao/datasets ]; then
    echo "Downloading datasets..."
    ## download using gdown
    mkdir -p datasets
    source ~/ids_venv/bin/activate
    python -m gdown "https://drive.google.com/uc?id=1gqnHB6kYAxyIFFHKtImHKVCXYYkZWKxI" -O datasets/dataset1.zip #AVTP/AEID
    python -m gdown "https://drive.google.com/uc?id=1nGybN0eAlgzYdHHDkRs4UGqKyCYmwXYO" -O datasets/dataset2.zip #TOW

    mkdir datasets/avtp-intrusion-dataset
    unzip datasets/dataset1.zip -d datasets/avtp-intrusion-dataset/raw
    mkdir datasets/tow-intrusion-dataset
    unzip datasets/dataset2.zip -d datasets/tow-intrusion-dataset/raw
    rm datasets/dataset1.zip datasets/dataset2.zip
    echo "Datasets downloaded and extracted."

    # copy to home auto ids evaluation framework directory
    cp -r datasets ~/automotive-ids-evaluation-framework-deteccao_intrusao/datasets 
else 
    echo "Datasets already downloaded."
fi

source ~/ids_venv/bin/activate

# sleep infinity

# feature generation
#python3 execute_feature_generator.py --feat_gen_config config_jsons/feat_generator/AVTP_CNNIDS_sumX_train.json ## for RF
python3 execute_feature_generator.py --feat_gen_config config_jsons/feat_generator/TOW_CNNIDS_Oneclass_train.json ## for RF

#python3 execute_feature_generator.py --feat_gen_config config_jsons/feat_generator/AVTP_CNNIDS_train.json ## for CNN
#python3 execute_feature_generator.py --feat_gen_config config_jsons/feat_generator/TOW_CNNIDS_Multiclass_train.json ## for CNN

# feature generator test 
#python3 execute_feature_generator.py --feat_gen_config config_jsons/feat_generator/AVTP_CNNIDS_sumX_test.json ## for RF
#python3 execute_feature_generator.py --feat_gen_config config_jsons/feat_generator/AVTP_CNNIDS_test.json ## for CNN
#python3 execute_feature_generator.py --feat_gen_config config_jsons/feat_generator/TOW_CNNIDS_Multiclass_test.json ## for CNN
#python3 execute_feature_generator.py --feat_gen_config config_jsons/feat_generator/TOW_CNNIDS_Oneclass_test.json ## for RF


# training and validation
#python3 execute_model_train_validation.py --model_train_valid_config config_jsons/model_train_validate/AVTP_RandomForest_train.json ## for RF
#python3 execute_model_train_validation.py --model_train_valid_config config_jsons/model_train_validate/TOW_RandomForest_train.json ## for RF
#python3 execute_model_train_validation.py --model_train_valid_config config_jsons/model_train_validate/AVTP_PrunedCNNIDS_train.json ## for CNN
#python3 execute_model_train_validation.py --model_train_valid_config config_jsons/model_train_validate/TOW_PrunedCNNIDS_Multiclass_train.json ## for CNN

# detection time evaluation
# python3 execute_model_test.py --model_test_config config_jsons/test_detection_time/AVTP_RandomForest_detection_time.json
# python3 execute_model_test.py --model_test_config config_jsons/test_detection_time/TOW_RandomForest_detection_time.json
# python3 execute_model_test.py --model_test_config config_jsons/test_detection_time/AVTP_PrunedCNNIDS_detection_time.json
# python3 execute_model_test.py --model_test_config config_jsons/test_detection_time/TOW_MC_PrunedCNNIDS_detection_time.json


# test evaluation
# python3 execute_model_test.py --model_test_config config_jsons/model_test/AVTP_MultiStage_test.json

# python3 execute_model_test.py --model_test_config config_jsons/model_test/TOW_MC_PrunedCNN_test.json
# python3 execute_model_test.py --model_test_config config_jsons/model_test/TOW_Oneclass_test.json


echo "Finish"   
