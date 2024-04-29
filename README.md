I would recommend starting by running `./setup.sh`, then work through `interp_measurement.ipynb`, which walks you through most of this work with commentary.

To train Chess SAEs, run the `setup.sh` script, then run `python circuits/train.py`. I'm currently loading the pytorch nanogpt state dict file into NNsight. I have ported the state dict into a GPT2LMHeadModel and stored it here: https://huggingface.co/adamkarvonen/8LayerChessGPT2
But I haven't yet switched my code over to using it and verifying that it produces equivalent results.

`setup.sh` will download and unzip some of my SAEs that I have on HuggingFace. If you have SAEs trained on Chess-GPT, you can place the folders (which contain `ae.pt` and `config.json`) into `autoencoders`. Or you can place them into `autoencoders/utils/` and run this: `rename_folder_for_config.ipynb` for descriptive folder names. It may also be required because I add some metadata to `config.json` there.

To perform analysis on a group of SAEs, run `python circuits/sae_stats_collection.py`. I have a preliminary hacky data analysis script in `analysis/sae_stats_analysis.ipynb`, with the results from the first group in `analysis/group1_results.json`.