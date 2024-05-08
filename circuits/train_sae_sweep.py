#%% 
# Imports
import torch as t
import numpy as np

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
sae_batch_size = 4096

num_tokens = 1_228_800_000 #300_000_000

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
# num_tokens = out_batch_size * steps
warmup_steps = 1000 # Warmup period at start of training and after each resample
save_steps = int(steps/4)
expansion_factor = 32
resample_steps = None
lr = 3e-4 # 3e-4
# initial_sparsity_penalties = t.logspace(np.log10(0.05), np.log10(5), 6)   

# for pythia 70m deduped
# initial_sparsity_penalties = list(t.logspace(-1.3,-1, 3)) + list(t.logspace(0.2, 0.5, 3))

# initial_sparsity_penalties = t.logspace(np.log10(0.003), np.log10(1), 6)    # for pythia 14m
# initial_sparsity_penalties = t.logspace(np.log10(0.001), np.log10(0.005), 10)    # for pythia 14m, jump
# initial_sparsity_penalties = t.logspace(np.log10(0.1), np.log10(1), 10)    # for pythia 14m, gated
# initial_sparsity_penalties = t.tensor([0.1,])
log_steps = 5 # Log the training 
p_start = 1
p_end = 0.2
anneal_start = int(steps/10) #10000
anneal_end = None #steps - int(steps/10)
#
trainer_configs = []
for sp in t.logspace(-1.4,-1.3, 3):
    sp = sp.item()
    trainer_configs.append({
        'trainer' : PAnnealTrainer,
        'dict_class' : AutoEncoder,
        'activation_dim' : d_submodule,
        'dict_size' : expansion_factor*d_submodule,
        'lr' : lr,
        'sparsity_function' : 'Lp^p',
        'initial_sparsity_penalty' : sp,
        'p_start' : p_start,
        'p_end' : p_end,
        'anneal_start' : anneal_start,
        'anneal_end' : anneal_end,
        'sparsity_queue_length' : 10,
        'n_sparsity_updates' : "continuous",
        'warmup_steps' : warmup_steps,
        'resample_steps' : resample_steps,
        'steps' : steps,
        'seed' : seed,
        'wandb_name' : f'PAnnealTrainer-chess-alpha{sp}L_p^p',
    })
#for sp in t.logspace(-1.1,-0.95, 3):
#    sp = sp.item()
#    trainer_configs.append({
#        'trainer' : StandardTrainer,
#        'dict_class' : AutoEncoder,
#        'activation_dim' : d_submodule,
#        'dict_size' : expansion_factor*d_submodule,
#        'lr' : lr,
#        'l1_penalty' : sp,
#        'warmup_steps' : warmup_steps,
#        'resample_steps' : resample_steps,
#        'seed' : seed,
#        'wandb_name' : f'StandardTrainer-chess-alpha{sp}',
#    })
#    
## for sp in t.logspace(-1.3,-1, 3):
##     sp = sp.item()
#    # trainer_configs.append({
#    #     'trainer' : JumpSAETrainer,
#    #     'dict_class' : JumpAutoEncoder,
#    #     'activation_dim' : d_submodule,
#    #     'dict_size' : expansion_factor*d_submodule,
#    #     'lr' : lr,
#    #     'l1_penalty' : sp,
#    #     'warmup_steps' : warmup_steps,
#    #     'resample_steps' : resample_steps,
#    #     'seed' : seed,
#    #     'wandb_name' : f'JumpSAETrainer-alpha{sp}',
#    # })
#
#for sp in t.logspace(-0.15, 0.15, 3):
#    sp = sp.item()
#    trainer_configs.append({
#        'trainer' : GatedSAETrainer,
#        'dict_class' : GatedAutoEncoder,
#        'activation_dim' : d_submodule,
#        'dict_size' : expansion_factor*d_submodule,
#        'lr' : lr,
#        'l1_penalty' : sp,
#        'warmup_steps' : warmup_steps,
#        'resample_steps' : resample_steps,
#        'seed' : seed,
#        'wandb_name' : f'GatedSAETrainer-chess-alpha{sp}',
#    })
#for sp in t.logspace(-0.4, -0.1, 3):
#    sp = sp.item()
#    trainer_configs.append({
#        'trainer' : GatedAnnealTrainer,
#        'dict_class' : GatedAutoEncoder,
#        'activation_dim' : d_submodule,
#        'dict_size' : expansion_factor*d_submodule,
#        'lr' : lr,
#        'sparsity_function' : 'Lp^p',
#        'initial_sparsity_penalty' : sp,
#        'p_start' : p_start,
#        'p_end' : p_end,
#        'anneal_start' : anneal_start,
#        'anneal_end' : anneal_end,
#        'sparsity_queue_length' : 10,
#        'n_sparsity_updates' : "continuous",
#        'warmup_steps' : warmup_steps,
#        'resample_steps' : resample_steps,
#        'steps' : steps,
#        'seed' : seed,
#        'wandb_name' : f'GatedAnnealTrainer-chess-alpha{sp}L_p^p',
#    })

print(f"len trainer configs: {len(trainer_configs)}")

save_dir = 'circuits/dictionary_learning/dictionaries/chess-p_anneal/'
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
