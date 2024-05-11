import torch
import itertools
import pickle

from circuits.nnsight_buffer import NNsightActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.standard import StandardTrainer

from circuits.utils import (
    chess_hf_dataset_to_generator,
    othello_hf_dataset_to_generator,
    get_model,
    get_submodule,
)

DEVICE = "cuda"

layer = 5
othello = False


if not othello:
    with open("models/meta.pkl", "rb") as f:
        meta = pickle.load(f)

    context_length = 256
    model_name = "adamkarvonen/8LayerChessGPT2"
    dataset_name = "adamkarvonen/chess_sae_text"
    data = chess_hf_dataset_to_generator(
        dataset_name, meta, context_length=context_length, split="train", streaming=True
    )
    model_type = "chess"
else:
    context_length = 59
    model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
    dataset_name = "taufeeque/othellogpt"
    data = othello_hf_dataset_to_generator(
        dataset_name, context_length=context_length, split="train", streaming=True
    )
    model_type = "othello"

model = get_model(model_name, DEVICE)
submodule = get_submodule(model_name, layer, model)

activation_dim = 512  # output dimension of the layer
resample_steps = 50000

buffer = NNsightActivationBuffer(
    data,
    model,
    submodule,
    n_ctxs=8e3,
    ctx_len=context_length,
    refresh_batch_size=128,
    io="out",
    d_submodule=activation_dim,
    device=DEVICE,
)

learning_rates = [1e-4, 1e-3]
l1_penalties = [1e-4, 1e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1]
dictionary_sizes = [4 * activation_dim, 16 * activation_dim]

param_combinations = list(itertools.product(learning_rates, l1_penalties, dictionary_sizes))

trainer_configs = [
    {
        "trainer": StandardTrainer,
        "lr": lr,
        "l1_penalty": l1,
        "warmup_steps": 2000,
        "resample_steps": resample_steps,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
    }
    for lr, l1, dictionary_size in param_combinations
]

trainSAE(
    buffer,
    trainer_configs=trainer_configs,
    steps=300000,
    save_steps=100000,
    save_dir=f"{model_type}_layer_{layer}",
    log_steps=1000,
)
