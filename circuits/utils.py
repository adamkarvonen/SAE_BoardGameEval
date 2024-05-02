from dataclasses import dataclass
import torch
from nnsight import LanguageModel
import json
from typing import Any
from datasets import load_dataset
from einops import rearrange
from jaxtyping import Int, Float, jaxtyped
from torch import Tensor

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
    model = LanguageModel(model, device_map=device, tokenizer=tokenizer).to(device)

    submodule = model.transformer.h[layer].mlp  # layer 1 MLP
    activation_dim = config["trainer"]["activation_dim"]  # output dimension of the MLP
    dictionary_size = config["trainer"]["dictionary_size"]

    buffer = ActivationBuffer(
        data,
        model,
        submodule,
        n_ctxs=512,
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


# TODO: This should take a list of dictionaries as input. Maybe in ae_bundle?
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
