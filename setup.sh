#!/bin/bash

pip install -r requirements.txt
pip install -e .
git submodule update --init

cd models
wget -O lichess_8layers_ckpt_no_optimizer.pt "https://huggingface.co/adamkarvonen/chess_llms/resolve/main/lichess_8layers_ckpt_no_optimizer.pt?download=true"

cd ..

cd circuits
cd dictionary_learning

git checkout collab
git pull