I just got this codebase working well today, so it isn't yet well documented. I hope to have things cleaned up and documented in the next day or two.

To train Chess SAEs, run the `setup.sh` script, then run `python circuits/train.py`. I'm currently loading the pytorch nanogpt state dict file into NNsight. I have ported the state dict into a GPT2LMHeadModel and stored it here: https://huggingface.co/adamkarvonen/8LayerChessGPT2
But I haven't yet switched my code over to using it and verifying that it produces equivalent results.

If you have SAEs trained on Chess-GPT, you can place the folders (which contain `ae.pt` and `config.json`) into `autoencoders`. Or you can place them into `autoencoders/utils/` and run this: `rename_folder_for_config.ipynb` for descriptive folder names. It may also be required because I add some metadata to `config.json` there. One of my other TODOs is to host my trained SAEs on HuggingFace and write a script to download them and organize them into my expected folder format.

Then, to perform analysis on SAEs, run `python circuits/sae_stats_collection.py`. There's also a Jupyter notebook that does things in a step by step way with partial commentary: `interp_measurement.ipynb`.