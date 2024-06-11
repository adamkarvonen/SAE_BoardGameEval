**Setup** 

Create a new virtual python environment (I'm using 3.11) and run `./setup.sh`, or run each command in `setup.sh` individually. This will install all requirements, installs the project in editable mode, and checks out the correct branch of the `dictionary_learning` submodule. By default, the repo includes an example Chess and Othello SAE. If you want to download sweeps of SAEs for analysis, refer to `autoencoders/download_saes.sh`.

**Getting Started**

There is a walkthrough of the approach in `circuits/full_pipeline_walkthrough.ipynb`.

To perform the analysis in the paper, run `python full_pipeline.py`. At the bottom of the script, you can select which autoencoder groups you want to analyze. The output of `full_pipeline.py` for the default autoencoder group is `f1_results.csv` at `autoencoders/testing_chess`.

This location of this `f1_results.csv` is already set in `f1_analysis.ipynb` to recreate all graphs from the paper. The data used to create the graphs from our paper can be found in `autoencoders/saved_data`.

**Shape Annotations**

I've been using this tip from Noam Shazeer:

Dimension key (from https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):

f = All SAE features

F = Batch of SAE features

B = batch_size

b = All inputs

L = seq length (context length)

T = thresholds

R = rows (or cols)

C = classes for one hot encoding

D = GPT d_model

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