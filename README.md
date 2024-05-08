I would recommend starting by running `./setup.sh`, then work through `analysis/classifier_analysis.ipynb`, which walks you through some of the results with some commentary. The evaluation is done with `circuits/eval_sae_as_classifier.py`.

To train Chess SAEs, run the `setup.sh` script, then run `python circuits/train.py`.

`setup.sh` will download and unzip some of my SAEs that I have on HuggingFace. If you have SAEs trained on Chess-GPT, you can place the folders (which contain `ae.pt` and `config.json`) into `autoencoders`. Or you can place them into `autoencoders/utils/` and run this: `rename_folder_for_config.ipynb` for descriptive folder names. It may also be required because I add some metadata to `config.json` there.

My SAEs and some saved eval results are here: https://huggingface.co/adamkarvonen/chess_saes

To perform analysis on a group of SAEs, run `python circuits/eval_sae_as_classifier.py`. At the bottom of the file, adjust the `custom_functions` list if needed. Note that there are "board states", such as `board_to_piece_state`, which contain the one hot state of the board at every character, or (832 elements = 8 rows x 8 cols x 13 possibilities per square). These run about 10x slower than higher level concepts such as `board_to_pin_state`, which only contain 1 element (binary value of is there a pin on the board).

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