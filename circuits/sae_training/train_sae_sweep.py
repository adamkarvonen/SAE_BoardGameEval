# %%
# Imports
import torch as t
import numpy as np
import itertools
import pickle

from nnsight import LanguageModel

from circuits.nanogpt_to_hf_transformers import NanogptTokenizer, convert_nanogpt_model
from circuits.utils import (
    chess_hf_dataset_to_generator,
    othello_hf_dataset_to_generator,
    get_model,
    get_submodule,
)

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.trainers.gated_anneal import GatedAnnealTrainer
from dictionary_learning.trainers.gdm import GatedSAETrainer
from dictionary_learning.trainers.jump import JumpSAETrainer
from dictionary_learning.trainers.standard_new import StandardTrainerNew
from dictionary_learning.trainers.p_anneal_new import PAnnealTrainerNew
from dictionary_learning.utils import hf_dataset_to_generator, zst_to_generator
from dictionary_learning.buffer import ActivationBuffer, NNsightActivationBuffer
from dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpAutoEncoder,
)

from IPython import embed

# %%
DEVICE = "cuda:0"

layer = 5
random_weights = False
othello = True

if not othello:
    with open("models/meta.pkl", "rb") as f:
        meta = pickle.load(f)

    context_length = 256
    if random_weights:
        model_name = "adamkarvonen/RandomWeights8LayerChessGPT2"
    else:
        model_name = "adamkarvonen/8LayerChessGPT2"
    dataset_name = "adamkarvonen/chess_sae_text"
    data = chess_hf_dataset_to_generator(
        dataset_name, meta, context_length=context_length, split="train", streaming=True
    )
    model_type = "chess"
else:
    context_length = 59
    if random_weights:
        model_name = "adamkarvonen/RandomWeights8LayerOthelloGPT2"
    else:
        model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
    dataset_name = "taufeeque/othellogpt"
    data = othello_hf_dataset_to_generator(
        dataset_name, context_length=context_length, split="train", streaming=True
    )
    model_type = "othello"

model = get_model(model_name, DEVICE)
submodule = get_submodule(model_name, layer, model)

activation_dim = 512  # output dimension of the layer

buffer_size = int(1e4 / 1)
llm_batch_size = 256
sae_batch_size = 8192

num_tokens = 300_000_000

activation_buffer = NNsightActivationBuffer(
    data,
    model,
    submodule,
    n_ctxs=8e3,
    ctx_len=256,
    refresh_batch_size=llm_batch_size,
    out_batch_size=sae_batch_size,
    io="out",
    d_submodule=activation_dim,
    device=DEVICE,
)


# %%
# Training

seed = 42
steps = int(num_tokens / sae_batch_size)  # Total number of batches to train
save_steps = int(steps / 4)


# constants for training
warmup_steps = 1000  # Warmup period at start of training and after each resample
resample_steps = None
log_steps = 5  # Log the training
p_start = 1
p_end = 0.2
anneal_end = None  # steps - int(steps/10)

# create the list of configs
trainer_configs = []

# grid search sweep
# old: learning_rate_ = t.logspace(start=-5, end=-2, steps=3, base=10)
expansion_factor_ = 2 ** t.arange(4, 5)
sparsity_queue_length_ = [10]
anneal_start_ = t.logspace(start=3.7, end=4.2, steps=3)
n_sparsity_updates_ = [10]
# old: initial_sparsity_penalty_ = t.logspace(-1.4,-1.1, 5)

# learning_rate_ = t.tensor([5e-6, 5e-5, 5e-4])
# initial_sparsity_penalty_ = t.logspace(-0.2231, 1.6094, 5) # [0.8, 5]
# param_combinations = itertools.product(learning_rate_, expansion_factor_, initial_sparsity_penalty_)
#
# for i, param_setting in enumerate(param_combinations):
#    lr, expansion_factor, sp = param_setting
#    trainer_configs.append({
#        'trainer' : StandardTrainerNew,
#        'dict_class' : AutoEncoderNew,
#        'activation_dim' : activation_dim,
#        'dict_size' : expansion_factor.item()*activation_dim,
#        'lr' : lr.item(),
#        'l1_penalty' : sp.item(),
#        'lambda_warm_steps' : warmup_steps,
#        'seed' : seed,
#        'wandb_name' : f'StandardTrainerNew-Anthropic-{"othello" if othello else "chess"}-{i}',
#    })


# learning_rate_ = t.tensor([5e-6, 5e-5, 5e-4])
# initial_sparsity_penalty_ = t.logspace(-0.2231, 1.6094, 5) # [0.8, 5]
# param_combinations = itertools.product(
#    learning_rate_,
#    expansion_factor_,
#    sparsity_queue_length_,
#    anneal_start_,
#    n_sparsity_updates_,
#    initial_sparsity_penalty_)
#
# for i, param_setting in enumerate(param_combinations):
#    lr, expansion_factor, sparsity_queue_length, anneal_start, n_sparsity_updates, sp = param_setting
#
#    trainer_configs.append({
#        'trainer' : PAnnealTrainerNew,
#        'dict_class' : AutoEncoderNew,
#        'activation_dim' : activation_dim,
#        'dict_size' : expansion_factor.item()*activation_dim,
#        'lr' : lr.item(),
#        'sparsity_function' : 'Lp^p',
#        'initial_sparsity_penalty' : sp.item(),
#        'p_start' : p_start,
#        'p_end' : p_end,
#        'anneal_start' : int(anneal_start.item()),
#        'anneal_end' : anneal_end,
#        'sparsity_queue_length' : sparsity_queue_length,
#        'n_sparsity_updates' : n_sparsity_updates,
#        'lambda_warm_steps' : warmup_steps,
#        'steps' : steps,
#        'seed' : seed,
#        'wandb_name' : f'PAnnealTrainerNew-Anthropic-{"othello" if othello else "chess"}-{i}',
#    })


# learning_rate_ = t.logspace(start=-5, end=-2, steps=3, base=10)
# initial_sparsity_penalty_ = t.logspace(-1.7,-1.2, 5)
# param_combinations = itertools.product(learning_rate_, expansion_factor_, initial_sparsity_penalty_)
#
# for i, param_setting in enumerate(param_combinations):
#    lr, expansion_factor, sp = param_setting
#    trainer_configs.append({
#        'trainer' : StandardTrainer,
#        'dict_class' : AutoEncoder,
#        'activation_dim' : activation_dim,
#        'dict_size' : expansion_factor.item()*activation_dim,
#        'lr' : lr.item(),
#        'l1_penalty' : sp.item(),
#        'warmup_steps' : warmup_steps,
#        'resample_steps' : resample_steps,
#        'seed' : seed,
#        'wandb_name' : f'StandardTrainer-othello-{i}',
#    })


# learning_rate_ = t.logspace(start=-5, end=-2, steps=3, base=10)
# initial_sparsity_penalty_ = t.logspace(-2.2,-1.5, 5)
# param_combinations = itertools.product(
#    learning_rate_,
#    expansion_factor_,
#    sparsity_queue_length_,
#    anneal_start_,
#    n_sparsity_updates_,
#    initial_sparsity_penalty_)
#
# for i, param_setting in enumerate(param_combinations):
#    lr, expansion_factor, sparsity_queue_length, anneal_start, n_sparsity_updates, sp = param_setting
#
#    trainer_configs.append({
#        'trainer' : PAnnealTrainer,
#        'dict_class' : AutoEncoder,
#        'activation_dim' : activation_dim,
#        'dict_size' : expansion_factor.item()*activation_dim,
#        'lr' : lr.item(),
#        'sparsity_function' : 'Lp^p',
#        'initial_sparsity_penalty' : sp.item(),
#        'p_start' : p_start,
#        'p_end' : p_end,
#        'anneal_start' : int(anneal_start.item()),
#        'anneal_end' : anneal_end,
#        'sparsity_queue_length' : sparsity_queue_length,
#        'n_sparsity_updates' : n_sparsity_updates,
#        'warmup_steps' : warmup_steps,
#        'resample_steps' : resample_steps,
#        'steps' : steps,
#        'seed' : seed,
#        'wandb_name' : f'PAnnealTrainer-othello-{i}',
#    })


# learning_rate_ = t.logspace(start=-5, end=-2, steps=3, base=10)
# initial_sparsity_penalty_ = t.logspace(-1.2, -0.8, 5)
# param_combinations = itertools.product(learning_rate_, expansion_factor_, initial_sparsity_penalty_)
#
# for i, param_setting in enumerate(param_combinations):
#
#    lr, expansion_factor, sp = param_setting
#    trainer_configs.append({
#        'trainer' : GatedSAETrainer,
#        'dict_class' : GatedAutoEncoder,
#        'activation_dim' : activation_dim,
#        'dict_size' : expansion_factor.item()*activation_dim,
#        'lr' : lr.item(),
#        'l1_penalty' : sp.item(),
#        'warmup_steps' : warmup_steps,
#        'resample_steps' : resample_steps,
#        'seed' : seed,
#        'wandb_name' : f'GatedSAETrainer-{"othello" if othello else "chess"}-{i}',
#    })


learning_rate_ = t.logspace(start=-5, end=-2, steps=3, base=10)
initial_sparsity_penalty_ = t.logspace(-1.3, -1.0, 5)
param_combinations = itertools.product(
    learning_rate_,
    expansion_factor_,
    sparsity_queue_length_,
    anneal_start_,
    n_sparsity_updates_,
    initial_sparsity_penalty_,
)

for i, param_setting in enumerate(param_combinations):
    lr, expansion_factor, sparsity_queue_length, anneal_start, n_sparsity_updates, sp = (
        param_setting
    )

    trainer_configs.append(
        {
            "trainer": GatedAnnealTrainer,
            "dict_class": GatedAutoEncoder,
            "activation_dim": activation_dim,
            "dict_size": expansion_factor.item() * activation_dim,
            "lr": lr.item(),
            "sparsity_function": "Lp^p",
            "initial_sparsity_penalty": sp.item(),
            "p_start": p_start,
            "p_end": p_end,
            "anneal_start": int(anneal_start.item()),
            "anneal_end": anneal_end,
            "sparsity_queue_length": sparsity_queue_length,
            "n_sparsity_updates": n_sparsity_updates,
            "warmup_steps": warmup_steps,
            "resample_steps": resample_steps,
            "steps": steps,
            "seed": seed,
            "wandb_name": f'GatedAnnealTrainer-{"othello" if othello else "chess"}-{i}',
        }
    )


print(f"len trainer configs: {len(trainer_configs)}")

save_dir = f'circuits/dictionary_learning/dictionaries/group-2024-05-18_{"othello" if othello else "chess"}-{"random_model" if random_weights else "trained_model"}-layer_{layer}-gated_anneal/'

print("save_dir: ", save_dir)

# %%
trainSAE(
    data=activation_buffer,
    trainer_configs=trainer_configs,
    steps=steps,
    save_steps=save_steps,
    save_dir=save_dir,
    log_steps=log_steps,
)
# %%
