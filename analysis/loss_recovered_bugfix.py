#%%
# imports


import torch
import numpy as np
import os
import sys
import pickle

home_dir = '/share/u/can'
repo_dir = '/share/u/can/chess-gpt-circuits'
sys.path.append(home_dir)
sys.path.append(repo_dir)
import circuits.chess_utils as chess_utils
import circuits.othello_utils as othello_utils
from dictionary_learning.buffer import ActivationBuffer, NNsightActivationBuffer

from circuits.eval_sae_as_classifier import (
    initialize_results_dict, 
    get_data_batch, 
    apply_indexing_function,
    construct_eval_dataset,
    construct_othello_dataset,
    prep_firing_rate_data,
)
from circuits.utils import (
    get_model, 
    get_submodule,
    get_ae_bundle,
    collect_activations_batch,
    get_nested_folders,
    get_firing_features,
    to_device,
    AutoEncoderBundle,
)

# Globals

# Dimension key (from https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
# F  = features and minibatch size depending on the context (maybe this is stupid)
# B = batch_size
# L = seq length (context length)
# T = thresholds
# R = rows (or cols)
# C = classes for one hot encoding

home_dir = '/share/u/can'
repo_dir = f'{home_dir}/chess-gpt-circuits'

DEVICE = 'cuda:0'
torch.set_grad_enabled(False)
batch_size = 32
feature_batch_size = batch_size
n_inputs = 2048 # Length of the eval dataset
GAME = "chess" # "chess" or "othello"

models_path = repo_dir + "/models/"

#%%
if GAME == "chess":
    othello = False

    autoencoder_group_paths = ["/autoencoders/group1/"]
    custom_functions = [chess_utils.board_to_piece_state] #, chess_utils.board_to_pin_state]
    model_name = "adamkarvonen/8LayerChessGPT2"
    # data = construct_eval_dataset(custom_functions, n_inputs, models_path=models_path, device=DEVICE)
    indexing_functions = [chess_utils.find_dots_indices]

elif GAME == "othello":
    othello = True

    autoencoder_group_paths = ["/autoencoders/othello_layer0/"]
    # autoencoder_group_paths = ["autoencoders/othello_layer0/", "autoencoders/othello_layer5_ef4/"]
    custom_functions = [
            # othello_utils.games_batch_no_last_move_to_state_stack_BLRRC,
            othello_utils.games_batch_to_state_stack_BLRRC,
            othello_utils.games_batch_to_state_stack_mine_yours_BLRRC,
        ]
    model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
    # data = construct_othello_dataset(custom_functions, n_inputs, models_path=models_path, device=DEVICE)
    indexing_functions = [chess_utils.get_othello_even_list_indices]  # I'm experimenting with these for Othello
else:
    raise ValueError("Invalid game")

all_autoencoder_paths = []
for group_path in autoencoder_group_paths:
    all_autoencoder_paths += get_nested_folders(repo_dir + group_path) 

#%%

n_ctxs = n_inputs
context_length = 128
activation_dim = 512
buffer_out_batch_size = 256

def gen(data):
    for sample in data['encoded_inputs']:
        yield sample


data = construct_eval_dataset(custom_functions, n_inputs, models_path=models_path, device=DEVICE)
generator = gen(data)

meta_path = models_path + "meta.pkl"
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
model = get_model(model_name, DEVICE)
submodule = get_submodule(model_name, layer=0, model=model)

vanilla_buffer = ActivationBuffer(
    generator,
    model,
    submodule,
    n_ctxs=n_ctxs,
    ctx_len=context_length,
    refresh_batch_size=buffer_out_batch_size,
    io="out",
    d_submodule=activation_dim,
    device=DEVICE,
    out_batch_size=buffer_out_batch_size,
)

modified_buffer = buffer = NNsightActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=n_ctxs,
        ctx_len=context_length,
        refresh_batch_size=buffer_out_batch_size,
        io="out",
        d_submodule=activation_dim,
        device=DEVICE,
        out_batch_size=buffer_out_batch_size,
    )
# %%

vb = vanilla_buffer.text_batch(batch_size=buffer_out_batch_size)
# %%
mb = modified_buffer.text_batch(batch_size=buffer_out_batch_size)
# %%

from dictionary_learning.evaluation import evaluate
from dictionary_learning import AutoEncoder

ae = AutoEncoder.from_pretrained(all_autoencoder_paths[0]+'ae.pt', device=DEVICE)

# %%
evaluate(
    dictionary=ae,
    activations=modified_buffer,
    device=DEVICE,
    batch_size=buffer_out_batch_size,
)
