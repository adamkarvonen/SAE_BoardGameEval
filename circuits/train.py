from nnsight import LanguageModel
import torch
import itertools

from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from circuits.nanogpt_to_hf_transformers import NanogptTokenizer, convert_nanogpt_model
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.trainers.standard import StandardTrainer

DEVICE = "cuda"

tokenizer = NanogptTokenizer("models/meta.pkl")
model = convert_nanogpt_model("models/lichess_8layers_ckpt_no_optimizer.pt", torch.device(DEVICE))
model = LanguageModel(model, device_map=DEVICE, tokenizer=tokenizer).to(DEVICE)

submodule = model.transformer.h[5]
activation_dim = 512  # output dimension of the layer
resample_steps = 50000


data = hf_dataset_to_generator("adamkarvonen/chess_sae_text")
buffer = ActivationBuffer(
    data,
    model,
    submodule,
    n_ctxs=8e3,
    ctx_len=256,
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
        "dictionary_size": dictionary_size,
    }
    for lr, l1, dictionary_size in param_combinations
]

trainSAE(
    buffer,
    trainer_configs=trainer_configs,
    steps=300000,
    save_steps=100000,
    save_dir="",
    log_steps=1000,
)
