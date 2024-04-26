import os
from nnsight import LanguageModel
import torch
import matplotlib.pyplot as plt
import chess
import json
import pickle

from circuits.dictionary_learning import ActivationBuffer
from circuits.nanogpt_to_hf_transformers import NanogptTokenizer, convert_nanogpt_model
from circuits.dictionary_learning.utils import hf_dataset_to_generator
from circuits.dictionary_learning import AutoEncoder
from circuits.dictionary_learning.interp import examine_dimension_chess

import circuits.chess_utils

DEVICE = torch.device("cuda")
MODEL_PATH = "models/lichess_8layers_ckpt_no_optimizer.pt"
batch_size = 8


def get_folders(path: str) -> list[str]:
    folder_names = []
    for folder_name in os.listdir(path):

        if not os.path.isdir(path + folder_name):
            continue

        if folder_name == "utils":
            continue

        folder_names.append(path + folder_name + "/")

    return folder_names


@torch.no_grad()
def get_feature(
    activations,
    ae: AutoEncoder,
    device,
) -> torch.Tensor:
    try:
        x = next(activations).to(device)
    except StopIteration:
        raise StopIteration(
            "Not enough activations in buffer. Pass a buffer with a smaller batch size or more data."
        )

    x_hat, f = ae(x, output_features=True)

    return f


def get_ae_stats(autoencoder_path: str) -> dict:

    autoencoder_model_path = f"{autoencoder_path}ae.pt"
    autoencoder_config_path = f"{autoencoder_path}config.json"
    ae = AutoEncoder.from_pretrained(autoencoder_model_path, device=DEVICE)

    with open(autoencoder_config_path, "r") as f:
        config = json.load(f)

    context_length = config["buffer"]["ctx_len"]
    layer = config["trainer"]["layer"]

    tokenizer = NanogptTokenizer()
    model = convert_nanogpt_model(MODEL_PATH, torch.device(DEVICE))
    model = LanguageModel(model, device_map=DEVICE, tokenizer=tokenizer).to(DEVICE)

    submodule = model.transformer.h[layer].mlp  # layer 1 MLP
    activation_dim = config["trainer"]["activation_dim"]  # output dimension of the MLP
    dictionary_size = config["trainer"]["dictionary_size"]

    # chess_sae_test is 100MB of data, so no big deal to download it
    data = hf_dataset_to_generator("adamkarvonen/chess_sae_test", streaming=False)
    buffer = ActivationBuffer(
        data,
        model,
        submodule,
        n_ctxs=512,
        ctx_len=context_length,
        refresh_batch_size=4,
        io="out",
        d_submodule=activation_dim,
        device=DEVICE,
        out_batch_size=batch_size,
    )

    total_inputs = 8192
    assert total_inputs % batch_size == 0
    num_iters = total_inputs // batch_size

    features = torch.zeros((total_inputs, dictionary_size), device=DEVICE)
    for i in range(num_iters):
        feature = get_feature(buffer, ae, DEVICE)  # (batch_size, dictionary_size)
        features[i * batch_size : (i + 1) * batch_size, :] = feature

    firing_rate_per_feature = (features != 0).float().sum(dim=0) / total_inputs

    assert firing_rate_per_feature.shape[0] == dictionary_size

    mask = (firing_rate_per_feature > 0) & (firing_rate_per_feature < 0.5)
    idx = torch.nonzero(mask, as_tuple=False).squeeze()

    per_dim_stats = examine_dimension_chess(
        model,
        submodule,
        buffer,
        dictionary=ae,
        dims=idx[:],
        n_inputs=1000,
        k=40,
        batch_size=25,
        device=DEVICE,
    )

    pickle.dump(per_dim_stats, open(f"{autoencoder_path}per_dim_stats.pkl", "wb"))

    return per_dim_stats


folders = get_folders("autoencoders/")

for folder in folders:
    get_ae_stats(folder)
    print(f"Finished {folder}")
