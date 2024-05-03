from dataclasses import dataclass
import torch
from nnsight import LanguageModel
import json
from typing import Any
from datasets import load_dataset
from einops import rearrange
from jaxtyping import Int, Float, jaxtyped
from torch import Tensor
import os
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from circuits.dictionary_learning import ActivationBuffer
from circuits.dictionary_learning import AutoEncoder
from circuits.nanogpt_to_hf_transformers import NanogptTokenizer, convert_nanogpt_model


@dataclass
class AutoEncoderBundle:
    ae: AutoEncoder
    buffer: ActivationBuffer
    model: LanguageModel
    activation_dim: int
    dictionary_size: int
    context_length: int
    submodule: Any


def get_ae_bundle(
    autoencoder_path: str,
    device: torch.device,
    data: Any,
    batch_size: int,
    model_path: str = "models/",
    n_ctxs: int = 512,
) -> AutoEncoderBundle:
    autoencoder_model_path = f"{autoencoder_path}ae.pt"
    autoencoder_config_path = f"{autoencoder_path}config.json"
    ae = AutoEncoder.from_pretrained(autoencoder_model_path, device=device)

    with open(autoencoder_config_path, "r") as f:
        config = json.load(f)

    context_length = config["buffer"]["ctx_len"]
    layer = config["trainer"]["layer"]

    tokenizer = NanogptTokenizer(meta_path=f"{model_path}meta.pkl")
    model = convert_nanogpt_model(
        f"{model_path}lichess_8layers_ckpt_no_optimizer.pt", torch.device(device)
    )
    # I'm pretty sure this will work, but I haven't tested it
    # model = GPT2LMHeadModel.from_pretrained("adamkarvonen/8LayerChessGPT2")
    model = LanguageModel(model, device_map=device, tokenizer=tokenizer).to(device)

    submodule = model.transformer.h[layer]  # residual stream after the layer
    activation_dim = config["trainer"]["activation_dim"]  # output dimension of the MLP
    dictionary_size = config["trainer"]["dictionary_size"]

    buffer = ActivationBuffer(
        data,
        model,
        submodule,
        n_ctxs=n_ctxs,
        ctx_len=context_length,
        refresh_batch_size=batch_size,
        io="out",
        d_submodule=activation_dim,
        device=device,
        out_batch_size=batch_size,
    )

    return AutoEncoderBundle(
        ae=ae,
        buffer=buffer,
        model=model,
        activation_dim=activation_dim,
        dictionary_size=dictionary_size,
        context_length=context_length,
        submodule=submodule,
    )


def get_first_n_dataset_rows(dataset_name: str, n: int, split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        count = 0
        for x in iter(dataset):
            if count >= n:
                break
            yield x["text"]
            count += 1

    return gen()


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


def get_firing_features(
    ae_bundle: AutoEncoderBundle,
    total_inputs: int,
    batch_size: int,
    device: torch.device,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Note: total inputs means the number of model activations, not the number of inputs to the model.
    total_inputs == n_inputs * context_length.
    For sparse autoencoders with larger expansion factors (16+), over 75% of the features can be dead.
    """

    num_iters = total_inputs // batch_size
    max_features = torch.full((ae_bundle.dictionary_size,), float("-inf"), device=device)

    features_F = torch.zeros((ae_bundle.dictionary_size,), device=device)
    for i in tqdm(range(num_iters), desc="Collecting features"):
        feature_BF = get_feature(ae_bundle.buffer, ae_bundle.ae, device)
        max_features = torch.max(max_features, feature_BF.max(dim=0).values)
        features_F += (feature_BF != 0).float().sum(dim=0)

    features_F /= total_inputs

    assert features_F.shape[0] == ae_bundle.dictionary_size

    mask = features_F > threshold

    alive_indices = torch.nonzero(mask, as_tuple=False).squeeze()
    max_features = max_features[alive_indices]

    return alive_indices, max_features


# TODO: This should take a list of dictionaries as input. Maybe in ae_bundle?
# On second thought, activation collection ends up being a relatively cheap operation
# compared to board state aggregation. I'll leave it as is for now.
@torch.no_grad()
def collect_activations_batch(
    ae_bundle: AutoEncoderBundle,
    inputs: list[str],
    dims: Int[Tensor, "num_dims"],
) -> tuple[Float[Tensor, "num_dims batch_size max_length"], Int[Tensor, "batch_size max_length"]]:
    with ae_bundle.model.trace(
        inputs, invoker_args=dict(max_length=ae_bundle.context_length, truncation=True)
    ):
        cur_tokens = ae_bundle.model.input[1][
            "input_ids"
        ].save()  # if you're getting errors, check here; might only work for pythia models
        cur_activations = ae_bundle.submodule.output
        if type(cur_activations.shape) == tuple:
            cur_activations = cur_activations[0]
        cur_activations = ae_bundle.ae.encode(cur_activations)
        cur_activations = cur_activations[
            :, :, dims
        ].save()  # Shape: (batch_size, max_length, dim_count)
    cur_activations = rearrange(
        cur_activations.value, "b n d -> d b n"
    )  # Shape: (dim_count, batch_size, max_length)

    return cur_activations, cur_tokens.value


def get_nested_folders(path: str) -> list[str]:
    """Get a list of folders nested one level deep in the given path which contain an ae.pt file"""
    folder_names = []
    # Process current directory and one level deep subdirectories
    for folder in os.listdir(path):
        if folder == "utils":
            continue
        current_folder = os.path.join(path, folder)
        if os.path.isdir(current_folder):
            if "ae.pt" in os.listdir(current_folder):
                folder_names.append(current_folder + "/")
            for subfolder in os.listdir(current_folder):  # Process subfolders
                subfolder_path = os.path.join(current_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    if "ae.pt" in os.listdir(subfolder_path):
                        folder_names.append(subfolder_path + "/")

    return folder_names
