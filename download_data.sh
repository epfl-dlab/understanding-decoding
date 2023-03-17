#!/bin/bash

# Unless the data_directory is passed as an argument download the files in the `data` directory
path_to_data_dir=${1:-data}

mkdir -p $path_to_data_dir
cd $path_to_data_dir

#################################
##### Download Pre-Trained Models
#################################
mkdir -p models
cd models

# Models initialized with a pretrained entity linker GENRE (GenIE - PLM)
wget https://zenodo.org/record/6139236/files/genie_genre_r.ckpt  # Trained on Rebel

cd ..
#################################

################
# Download Tries
################
wget https://zenodo.org/record/6139236/files/tries.zip
unzip tries.zip && rm tries.zip
###################

###########################
 Download World Definitions
###########################
wget https://zenodo.org/record/6139236/files/world_definitions.zip
unzip world_definitions.zip && rm world_definitions.zip
##################

#####################################
# Download Noisy Oracles for Toxicity
#####################################
mkdir -p detoxify/noisy_oracles
cd detoxify/noisy_oracles

wget "https://huggingface.co/martinjosifoski/detoxify_noisy_oracles/resolve/main/N-Step-Checkpoint_epoch%3D0_global_step%3D200.ckpt"
wget "https://huggingface.co/martinjosifoski/detoxify_noisy_oracles/resolve/main/N-Step-Checkpoint_epoch%3D0_global_step%3D400.ckpt"
wget "https://huggingface.co/martinjosifoski/detoxify_noisy_oracles/resolve/main/N-Step-Checkpoint_epoch%3D0_global_step%3D800.ckpt"
wget "https://huggingface.co/martinjosifoski/detoxify_noisy_oracles/resolve/main/N-Step-Checkpoint_epoch%3D0_global_step%3D1200.ckpt"
#######################################

echo "The data was download at '$path_to_data_dir'."
