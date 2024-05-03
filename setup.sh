#!/bin/bash

# Check if unzip is installed, and exit if it isn't
if ! command -v unzip &> /dev/null
then
    echo "Error: unzip is not installed. Please install it and rerun the setup script."
    exit 1
fi

pip install -r requirements.txt
pip install -e .
git submodule update --init

cd models
wget -O lichess_8layers_ckpt_no_optimizer.pt "https://huggingface.co/adamkarvonen/chess_llms/resolve/main/lichess_8layers_ckpt_no_optimizer.pt?download=true"

cd ..

cd autoencoders

wget -O group1.zip "https://huggingface.co/adamkarvonen/chess_saes/resolve/main/group1.zip?download=true"
unzip group1.zip
rm group1.zip

cd ..

cd analysis

wget -O group1_results.zip "https://huggingface.co/adamkarvonen/chess_saes/resolve/main/group1_results.zip?download=true"
unzip group1_results.zip
rm group1_results.zip

cd circuits
cd dictionary_learning

git checkout collab
git pull