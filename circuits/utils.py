from dataclasses import dataclass
import torch
from nnsight import NNsight
import json
from typing import Any
from datasets import load_dataset
from einops import rearrange
from jaxtyping import Int, Float, jaxtyped
from torch import Tensor
import os
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from transformer_lens import HookedTransformer

from circuits.dictionary_learning.buffer import NNsightActivationBuffer
from circuits.dictionary_learning import AutoEncoder
from circuits.chess_utils import encode_string
from circuits.dictionary_learning import ActivationBuffer
from circuits.dictionary_learning.dictionary import AutoEncoder, GatedAutoEncoder, AutoEncoderNew
from circuits.dictionary_learning.trainers.gated_anneal import GatedAnnealTrainer
from circuits.dictionary_learning.trainers.gdm import GatedSAETrainer
from circuits.dictionary_learning.trainers.p_anneal import PAnnealTrainer
from circuits.dictionary_learning.trainers.p_anneal_new import PAnnealTrainerNew
from circuits.dictionary_learning.trainers.standard import StandardTrainer
from circuits.dictionary_learning.trainers.standard_new import StandardTrainerNew
from circuits.nanogpt_to_hf_transformers import NanogptTokenizer, convert_nanogpt_model

from IPython import embed


@dataclass
class AutoEncoderBundle:
    ae: AutoEncoder
    buffer: NNsightActivationBuffer
    model: NNsight
    activation_dim: int
    dictionary_size: int
    context_length: int
    submodule: Any


def get_model(model_name: str, device: torch.device) -> NNsight:
    if model_name == "Baidicoot/Othello-GPT-Transformer-Lens":
        tf_model = HookedTransformer.from_pretrained("Baidicoot/Othello-GPT-Transformer-Lens")
        model = NNsight(tf_model).to(device)
        return model

    if (
        model_name == "adamkarvonen/RandomWeights8LayerOthelloGPT2"
        or model_name == "adamkarvonen/RandomWeights8LayerChessGPT2"
    ):
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        model = NNsight(model).to(device)
        return model

    if model_name == "adamkarvonen/8LayerChessGPT2":
        # Old method of loading model from nanogpt weights
        # model = convert_nanogpt_model(
        #     f"{model_path}lichess_8layers_ckpt_no_optimizer.pt", torch.device(device)
        # )
        # tokenizer = NanogptTokenizer(meta_path=f"{model_path}meta.pkl")
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        model = NNsight(model).to(device)
        return model

    raise ValueError("Model not found.")


def get_submodule(model_name: str, layer: int, model: NNsight) -> Any:
    if model_name == "Baidicoot/Othello-GPT-Transformer-Lens":
        return model.blocks[layer].hook_resid_post
    if model_name == "adamkarvonen/8LayerChessGPT2":
        return model.transformer.h[layer]  # residual stream after the layer
    raise ValueError("Model not found.")


def get_ae_bundle(
    autoencoder_path: str,
    device: torch.device,
    data: Any,  # iter of list of ints
    batch_size: int,
    model_path: str = "models/",
    model_name: str = "adamkarvonen/8LayerChessGPT2",
    n_ctxs: int = 512,
) -> AutoEncoderBundle:
    autoencoder_model_path = f"{autoencoder_path}ae.pt"
    autoencoder_config_path = f"{autoencoder_path}config.json"

    with open(autoencoder_config_path, "r") as f:
        config = json.load(f)

    # rangell: this is a super hacky way to get the correct dictionary class from the config
    # TODO (rangell): make this better in the future.
    # old: ae = AutoEncoder.from_pretrained(autoencoder_model_path, device=device)
    config_args = []
    for k, v in config["trainer"].items():
        if k not in ["trainer_class", "sparsity_penalty"]:
            if isinstance(v, str) and k != "dict_class":
                config_args.append(k + "=" + "'" + v + "'")
            else:
                config_args.append(k + "=" + str(v))
    config_str = ", ".join(config_args)
    ae = eval(config["trainer"]["trainer_class"] + f"({config_str})").ae.__class__.from_pretrained(
        autoencoder_model_path, device=device
    )

    context_length = config["buffer"]["ctx_len"]
    # layer = config["trainer"]["layer"]
    layer = 5
    print(f"WARNING: using manual setting of layer to {layer}")

    model = get_model(model_name, device)
    submodule = get_submodule(model_name, layer, model)

    activation_dim = ae.activation_dim
    dictionary_size = ae.dict_size

    buffer = NNsightActivationBuffer(
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

    alive_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    max_features = max_features[alive_indices]

    return alive_indices, max_features


# TODO: This should take a list of dictionaries as input. Maybe in ae_bundle?
# On second thought, activation collection ends up being a relatively cheap operation
# compared to board state aggregation. I'll leave it as is for now.
@torch.no_grad()
def collect_activations_batch(
    ae_bundle: AutoEncoderBundle,
    inputs_BL: torch.Tensor,
    dims: Int[Tensor, "num_dims"],
) -> tuple[Float[Tensor, "num_dims batch_size max_length"], Int[Tensor, "batch_size max_length"]]:
    with ae_bundle.model.trace(
        inputs_BL, invoker_args=dict(max_length=ae_bundle.context_length, truncation=True)
    ):
        cur_tokens = ae_bundle.model.input[0].save()
        cur_activations = ae_bundle.submodule.output
        if type(cur_activations.shape) == tuple:
            cur_activations = cur_activations[0]
        cur_activations = ae_bundle.ae.encode(cur_activations)
        cur_activations_BLF = cur_activations[
            :, :, dims
        ].save()  # Shape: (batch_size, max_length, dim_count)
    cur_activations_FBL = rearrange(
        cur_activations_BLF.value, "b n d -> d b n"
    )  # Shape: (dim_count, batch_size, max_length)

    return cur_activations_FBL, cur_tokens.value[0]


@torch.no_grad()
def get_model_activations(
    ae_bundle: AutoEncoderBundle,
    inputs_BL: torch.Tensor,
) -> tuple[Float[Tensor, "num_dims batch_size max_length"], Int[Tensor, "batch_size max_length"]]:
    with ae_bundle.model.trace(
        inputs_BL, invoker_args=dict(max_length=ae_bundle.context_length, truncation=True)
    ):
        cur_activations = ae_bundle.submodule.output
        if type(cur_activations.shape) == tuple:
            cur_activations = cur_activations[0].save()
    return cur_activations


@torch.no_grad()
def get_feature_activations_batch(
    ae_bundle: AutoEncoderBundle,
    model_activations_BLD: torch.Tensor,
    dims: Int[Tensor, "num_dims"],
) -> tuple[Float[Tensor, "num_dims batch_size max_length"], Int[Tensor, "batch_size max_length"]]:
    feature_activations = ae_bundle.ae.encode(model_activations_BLD)
    feature_activations_BLF = feature_activations[:, :, dims]
    feature_activations_FBL = rearrange(
        feature_activations_BLF, "b n d -> d b n"
    )  # Shape: (dim_count, batch_size, max_length)
    return feature_activations_FBL


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


def to_device(data, device):
    """
    Recursively move tensors in a nested dictionary to CPU.
    """
    if isinstance(data, dict):
        # If it's a dictionary, apply recursively to each value
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        # If it's a list, apply recursively to each element
        return [to_device(item, device) for item in data]
    elif isinstance(data, torch.Tensor):
        # If it's a tensor, move it to CPU
        return data.to(device)
    else:
        # If it's neither, return it as is
        return data


def othello_hf_dataset_to_generator(
    dataset_name: str, context_length: int = 59, split="train", streaming=True
):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            yield x["tokens"][:context_length]

    return gen()


def chess_hf_dataset_to_generator(
    dataset_name: str, meta: dict, context_length: int = 256, split="train", streaming=True
):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            yield encode_string(meta, x["text"][:context_length])

    return gen()
