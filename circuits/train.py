from nnsight import LanguageModel
import torch

from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from circuits.nanogpt_to_hf_transformers import NanogptTokenizer, convert_nanogpt_model
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.trainers.standard import StandardTrainer

DEVICE = "cuda"

tokenizer = NanogptTokenizer("models/meta.pkl")
model = convert_nanogpt_model("models/lichess_8layers_ckpt_no_optimizer.pt", torch.device(DEVICE))
model = LanguageModel(model, device_map=DEVICE, tokenizer=tokenizer).to(DEVICE)

submodule = model.transformer.h[5].mlp  # layer 1 MLP
activation_dim = 512  # output dimension of the MLP
dictionary_size = 16 * activation_dim


data = hf_dataset_to_generator("adamkarvonen/chess_sae_text")
buffer = ActivationBuffer(
    data,
    model,
    submodule,
    n_ctxs=8e3,
    ctx_len=256,
    refresh_batch_size=128,
    io="out",
    d_submodule=512,
    device=DEVICE,
)

trainSAE(
    buffer,
    activation_dim=activation_dim,
    dictionary_size=dictionary_size,
    trainer_configs=[
        {
            "trainer": StandardTrainer,
            "lr": 1e-4,
            "l1_penalty": 1e-2,
            "warmup_steps": 2000,
            "resample_steps": 150000,
        },
        {
            "trainer": StandardTrainer,
            "lr": 1e-4,
            "l1_penalty": 3e-2,
            "warmup_steps": 2000,
            "resample_steps": 150000,
        },
        {
            "trainer": StandardTrainer,
            "lr": 1e-4,
            "l1_penalty": 6e-2,
            "warmup_steps": 2000,
            "resample_steps": 150000,
        },
        {
            "trainer": StandardTrainer,
            "lr": 1e-4,
            "l1_penalty": 1e-1,
            "warmup_steps": 2000,
            "resample_steps": 150000,
        },
        {
            "trainer": StandardTrainer,
            "lr": 1e-4,
            "l1_penalty": 3e-1,
            "warmup_steps": 2000,
            "resample_steps": 150000,
        },
        {
            "trainer": StandardTrainer,
            "lr": 1e-3,
            "l1_penalty": 1e-2,
            "warmup_steps": 2000,
            "resample_steps": 150000,
        },
        {
            "trainer": StandardTrainer,
            "lr": 1e-3,
            "l1_penalty": 3e-2,
            "warmup_steps": 2000,
            "resample_steps": 150000,
        },
        {
            "trainer": StandardTrainer,
            "lr": 1e-3,
            "l1_penalty": 6e-2,
            "warmup_steps": 2000,
            "resample_steps": 150000,
        },
        {
            "trainer": StandardTrainer,
            "lr": 1e-3,
            "l1_penalty": 1e-1,
            "warmup_steps": 2000,
            "resample_steps": 150000,
        },
        {
            "trainer": StandardTrainer,
            "lr": 1e-3,
            "l1_penalty": 3e-1,
            "warmup_steps": 2000,
            "resample_steps": 150000,
        },
    ],
    steps=300000,
    save_steps=150000,
    save_dir="",
    log_steps=1000,
)
