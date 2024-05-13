#%% 
# Imports
import torch as t
import numpy as np
import itertools

from nnsight import LanguageModel

from circuits.nanogpt_to_hf_transformers import NanogptTokenizer, convert_nanogpt_model

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.trainers.gated_anneal import GatedAnnealTrainer
from dictionary_learning.trainers.gdm import GatedSAETrainer
from dictionary_learning.trainers.jump import JumpSAETrainer
from dictionary_learning.trainers.standard_new import StandardTrainerNew
from dictionary_learning.utils import hf_dataset_to_generator, zst_to_generator
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder, GatedAutoEncoder, AutoEncoderNew, JumpAutoEncoder

#%% 
DEVICE = 'cuda:0'

# load chess-gpt
tokenizer = NanogptTokenizer("models/meta.pkl")
model = convert_nanogpt_model("models/lichess_8layers_ckpt_no_optimizer.pt", t.device(DEVICE))
model = LanguageModel(model, device_map=DEVICE, tokenizer=tokenizer).to(DEVICE)
submodule = model.transformer.h[5]
d_submodule = model.config.hidden_size

buffer_size = int(1e4/1)
llm_batch_size = 256
sae_batch_size = 8192

num_tokens = 300_000_000

# load chess data
generator = hf_dataset_to_generator("adamkarvonen/chess_sae_text")
activation_buffer = ActivationBuffer(
    data=generator,
    model=model,
    submodule=submodule,
    d_submodule=d_submodule,
    io='out',
    n_ctxs=buffer_size,
    ctx_len=256,
    refresh_batch_size=llm_batch_size, # batches for buffer internal activations
    out_batch_size=sae_batch_size, # batches for training
    device=DEVICE
)

#%% 
# Training

seed = 42
steps = int(num_tokens / sae_batch_size) # Total number of batches to train
save_steps = int(steps/4)


# constants for training
warmup_steps = 1000 # Warmup period at start of training and after each resample
resample_steps = None
log_steps = 5 # Log the training 
p_start = 1
p_end = 0.2
anneal_end = None #steps - int(steps/10)

# create the list of configs
trainer_configs = []

# grid search sweep
learning_rate_ = t.logspace(start=-5, end=-2, steps=3, base=10)
expansion_factor_ = 2**t.arange(3, 5)
sparsity_queue_length_ = [10]
anneal_start_ = t.logspace(start=2, end=4, steps=2, base=10)
n_sparsity_updates_ = [10]
initial_sparsity_penalty_ = t.logspace(-1.7,-1.2, 5)

param_combinations = itertools.product(
    learning_rate_,
    expansion_factor_,
    sparsity_queue_length_,
    anneal_start_,
    n_sparsity_updates_,
    initial_sparsity_penalty_)
    
for i, param_setting in enumerate(param_combinations):
    lr, expansion_factor, sparsity_queue_length, anneal_start, n_sparsity_updates, sp = param_setting

    trainer_configs.append({
        'trainer' : PAnnealTrainer,
        'dict_class' : AutoEncoder,
        'activation_dim' : d_submodule,
        'dict_size' : expansion_factor.item()*d_submodule,
        'lr' : lr.item(),
        'sparsity_function' : 'Lp^p',
        'initial_sparsity_penalty' : sp.item(),
        'p_start' : p_start,
        'p_end' : p_end,
        'anneal_start' : int(anneal_start.item()),
        'anneal_end' : anneal_end,
        'sparsity_queue_length' : sparsity_queue_length,
        'n_sparsity_updates' : n_sparsity_updates,
        'warmup_steps' : warmup_steps,
        'resample_steps' : resample_steps,
        'steps' : steps,
        'seed' : seed,
        'wandb_name' : f'PAnnealTrainer-chess-{i}',
    })


#param_combinations = itertools.product(learning_rate_, expansion_factor_, initial_sparsity_penalty_)
#
#for i, param_setting in enumerate(param_combinations):
#    lr, expansion_factor, sp = param_setting
#    trainer_configs.append({
#        'trainer' : StandardTrainer,
#        'dict_class' : AutoEncoder,
#        'activation_dim' : d_submodule,
#        'dict_size' : expansion_factor.item()*d_submodule,
#        'lr' : lr.item(),
#        'l1_penalty' : sp.item(),
#        'warmup_steps' : warmup_steps,
#        'resample_steps' : resample_steps,
#        'seed' : seed,
#        'wandb_name' : f'StandardTrainer-chess-{i}',
#    })
    


#param_combinations = itertools.product(learning_rate_, expansion_factor_, initial_sparsity_penalty_)
#
#for i, param_setting in enumerate(param_combinations):
#    lr, expansion_factor, sp = param_setting
#    trainer_configs.append({
#        'trainer' : GatedSAETrainer,
#        'dict_class' : GatedAutoEncoder,
#        'activation_dim' : d_submodule,
#        'dict_size' : expansion_factor.item()*d_submodule,
#        'lr' : lr.item(),
#        'l1_penalty' : sp.item(),
#        'warmup_steps' : warmup_steps,
#        'resample_steps' : resample_steps,
#        'seed' : seed,
#        'wandb_name' : f'GatedSAETrainer-chess-{i}',
#    })


#for i, param_setting in enumerate(param_combinations):
#    lr, expansion_factor, sparsity_queue_length, anneal_start, n_sparsity_updates, sp = param_setting
#
#    trainer_configs.append({
#        'trainer' : GatedAnnealTrainer,
#        'dict_class' : GatedAutoEncoder,
#        'activation_dim' : d_submodule,
#        'dict_size' : expansion_factor.item()*d_submodule,
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
#        'wandb_name' : f'GatedAnnealTrainer-chess-{i}',
#    })

print(f"len trainer configs: {len(trainer_configs)}")

save_dir = 'circuits/dictionary_learning/dictionaries/group-2024-05-11_chess-p_anneal/'
#%%
trainSAE(
    data = activation_buffer, 
    trainer_configs = trainer_configs,
    steps=steps,
    save_steps=save_steps,
    save_dir=save_dir,
    log_steps=log_steps,  
)
# %%
