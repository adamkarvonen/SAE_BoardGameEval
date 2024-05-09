I would recommend starting by running `./setup.sh`, then work through `analysis/classifier_analysis.ipynb`, which walks you through some of the results with some commentary. I do need to add some better commentary, but it is something. The evaluation is done with `circuits/eval_sae_as_classifier.py`.

To train Chess SAEs, run the `setup.sh` script, then run `python circuits/train.py`. At the top of `train.py`, set the bool for Othello (if False, it will train on chess) and set the layer number. The train script should save the results into a folder. Drag that folder into `autoencoders/utils/`, set the layer number in `rename_folder_for_config.ipynb`, and run the Jupyter notebook. This is primarily necessary to add the layer number to `config.json` in every autoencoder.

If you don't want to train your own, `setup.sh` will download and unzip some of my SAEs that I have on HuggingFace. Most of them have been modified by `rename_folder_for_config.ipynb` already. If you get errors about the layer key, you will need to run that script.

Once you have trained SAEs, drag the folder containing the SAEs into `autoencoders`. It should look something like `autoencoders/group1/`, where `group1` contains many SAE folders, and every SAE folder contains `ae.pt` and `config.json`. Then, set this at the bottom of `circuits/eval_sae_as_classifier.py` to equal your autoencoder group path: `autoencoder_group_path = "autoencoders/group1/"`. If you want to test out additional custom functions, add them to `custom_functions`. Also set the `othello` bool (TODO: Add this to SAE configs somewhere). Then run `python circuits/eval_sae_as_classifier.py`. For 1000 n_inputs, and ~1000 alive features in the SAE, it takes around 2 minutes on a RTX 3090.

Then, put all of the results files (which will be in the repo root at `chess-gpt-circuits/`) into a folder, and place that folder into `analysis/`. If you don't want to evaluate your own, `setup.sh` will download and unzip some of my results that I have on HuggingFace. You can place these, such as `group1_results/`, into `analysis/`. My SAEs and some saved eval results are here: https://huggingface.co/adamkarvonen/chess_saes

Then, at the bottom of `analysis.py` set `folder_name = "group1_results/"` to equal your folder name, change directory into `analysis/`, and run `python analysis.py`. It will display the results in the terminal.

Note on `eval_sae_as_classifier.py`: there are "board states", such as `board_to_piece_state`, which contain the one hot state of the board at every character, or (832 elements = 8 rows x 8 cols x 13 possibilities per square). These run about 10x slower than higher level concepts such as `board_to_pin_state`, which only contain 1 element (binary value of is there a pin on the board).

To analyze 1000 inputs on a SAE with 1200 alive features on `board_to_pin_state` on an RTX 3090, it takes 11 seconds. For `board_to_pin_state` and `board_to_piece_state`, it's 140 seconds.

**OUTDATED**
To perform analysis on a group of SAEs, run `python circuits/sae_stats_collection.py`. I have a preliminary hacky data analysis script in `analysis/sae_stats_analysis.ipynb`, with the results from the first group in `analysis/group1_results.json`.

**Shape Annotations**

I've been using this tip from Noam Shazeer:

Dimension key (from https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):

F  = features and minibatch size depending on the context (maybe this is stupid)

B = batch_size

L = seq length (context length)

T = thresholds

R = rows (or cols)

C = classes for one hot encoding

For example, boards begins as shape (Batch, seq_len, rows, rows, classes), and after einops.repeat is shape (num_thresholds, num_features, batch_size, seq_len, rows, rows, classes).


```
boards_BLRRC = batch_data[custom_function.__name__]
boards_TFBLRRC = einops.repeat(
    boards_BLRRC,
    "B L R1 R2 C -> T F B L R1 R2 C",
    F=f_batch_size,
    T=len(thresholds_T111),
)
```