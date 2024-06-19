#!/bin/bash

pip install -r requirements.txt
pip install -e .
git submodule update --init

cd circuits
cd dictionary_learning
git pull
git checkout collab
# confirmed working with dictionary_learning commit 113c042101b6df6de60b04c7e65116c3a9460904
git pull