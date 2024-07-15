**Setup** 

Create a new virtual python environment (I'm using 3.11). Then, run:

```
pip install -r requirements.txt
pip install -e .
git submodule update --init
```

This will install all requirements, installs the project in editable mode, and install the correct commit of the `dictionary_learning` submodule. By default, the repo includes an example Chess and Othello SAE. If you want to download sweeps of SAEs for analysis, refer to `autoencoders/download_saes.sh`.

**Getting Started**

There is a walkthrough of the approach in `circuits/full_pipeline_walkthrough.ipynb`.

To perform the analysis in the paper, run `python full_pipeline.py`. By default, it runs on the single Chess SAE in `autoencoders/testing_chess`. It takes a few minutes on an RTX 3090 and uses < 10GB of VRAM. By decreasing the batch size, it can run using < 2 GB of VRAM if necessary. At the bottom of the script, you can select which autoencoder groups you want to analyze. The output of `full_pipeline.py` for the default autoencoder group is `f1_results.csv` at `autoencoders/testing_chess`.

The `full_pipeline` can be ran on SAE feature activations or MLP neuron activations on both ChessGPT and OthelloGPT. You just have to select the autoencoder group path, and everything else should happen automatically. Refer to `circuits/pipeline_config.py` to set config values and for explanations of their purpose. To decrease runtime, we support parallel analysis on multiple GPUs. This can also be set in `pipeline_config.py`.

`f1_analysis.ipynb` is used to create all graphs in the paper. The data used to create the graphs from our paper can be found in `autoencoders/saved_data`.By default, the csv path in `f1_analysis.ipynb` is set to the saved chess data, and you can recreate all paper chess graphs by running the notebook. New results can be analyzed by updating the csv path in `f1_analysis.ipynb`.

**SAE Training**

To train SAEs on ChessGPT or OthelloGPT, refer to the README in `circuits/sae_training`.

**Shape Annotations**

I've been using this tip from Noam Shazeer:

Dimension key (from https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):

f = All SAE features

F = Batch of SAE features

b = All inputs

B = batch_size

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

**Tests**

Run `pytest -s` from the root directory to run all tests. This will take a couple minutes, and `-s` is helpful to gauge progress.