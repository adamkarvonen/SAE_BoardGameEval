from dataclasses import dataclass
import torch
from nnsight import LanguageModel
import json
from typing import Any

from circuits.dictionary_learning import ActivationBuffer
from circuits.dictionary_learning.utils import hf_dataset_to_generator
from circuits.dictionary_learning import AutoEncoder
from circuits.dictionary_learning.evaluation import evaluate

from circuits.nanogpt_to_hf_transformers import NanogptTokenizer, convert_nanogpt_model
from circuits.chess_interp import examine_dimension_chess
import circuits.chess_utils as chess_utils
import circuits.chess_interp as chess_interp


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
    device: str,
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
