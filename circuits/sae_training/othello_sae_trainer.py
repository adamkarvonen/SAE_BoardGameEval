# Simplified version of train saes parallel
# TODO parallelize training for SAEs on multiple layers

import argparse
import torch as t
import gc

from circuits.utils import (
    othello_hf_dataset_to_generator,
    get_model,
)

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.buffer import NNsightActivationBuffer
from dictionary_learning.dictionary import AutoEncoder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True,
                        help="where to store sweep")
    parser.add_argument("--dry_run", action="store_true",  help="dry run sweep")
    args = parser.parse_args()
    return args


def run_sae_training(
        layer : int,
        save_dir : str,
        device : str,
        dry_run : bool = False
):
   
    # model and data parameters
    model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
    dataset_name = "taufeeque/othellogpt"
    context_length = 59
    activation_dim = 512 * 4  # output dimension of the layer

    buffer_size = int(3e4 / 4)
    llm_batch_size = 128 # 256 for A100 GPU, 64 for 1080ti
    sae_batch_size = 8192
    num_tokens = 500_000_000

    # sae training parameters
    seed = 42
    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train
    save_steps = int(steps / 4)
    warmup_steps = 1000  # Warmup period at start of training and after each resample
    resample_steps = None
    log_steps = 10  # Log the training
    p_start = 1
    p_end = 0.2
    anneal_end = None  # steps - int(steps/10)
    learning_rate = 0.0003162277571391314
    expansion_factor = 8
    sparsity_queue_length = 10
    anneal_start = 1000
    n_sparsity_updates = 10
    initial_sparsity_penalty = 0.03162277489900589   


    # Initialize model, data and activation buffer
    model = get_model(model_name, device)
    # submodule = model.blocks[layer].hook_resid_post # resid_post
    submodule = model.blocks[layer].mlp.hook_post # resid_pre
    data = othello_hf_dataset_to_generator(
        dataset_name, context_length=context_length, split="train", streaming=True
    )
    activation_buffer = NNsightActivationBuffer(
        data,
        model,
        submodule,
        n_ctxs=buffer_size,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io="out",
        d_submodule=activation_dim,
        device=device,
    )

    # create the list of configs
    trainer_configs = []
    trainer_configs.append(
        {
            "trainer": PAnnealTrainer,
            "dict_class": AutoEncoder,
            "activation_dim": activation_dim,
            "dict_size": expansion_factor * activation_dim,
            "lr": learning_rate,
            "sparsity_function": "Lp^p",
            "initial_sparsity_penalty": initial_sparsity_penalty,
            "p_start": p_start,
            "p_end": p_end,
            "anneal_start": int(anneal_start),
            "anneal_end": anneal_end,
            "sparsity_queue_length": sparsity_queue_length,
            "n_sparsity_updates": n_sparsity_updates,
            "warmup_steps": warmup_steps,
            "resample_steps": resample_steps,
            "steps": steps,
            "seed": seed,
            "wandb_name": f"PAnnealTrainer-othello-mlp-layer-{layer}",
            "layer" : layer,
            "lm_name" : model_name,
            "device": device,
        }
    )


    print(f"len trainer configs: {len(trainer_configs)}")
    save_dir = f'{save_dir}layer_{layer}'

    if not dry_run:
        # actually run the sweep
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
        )

if __name__ == "__main__":
    args = get_args()
    for layer in range(8):
        run_sae_training(
            layer=layer,
            save_dir=args.save_dir,
            device="cuda:0",
            dry_run=args.dry_run,
        )
        t.cuda.empty_cache()
        gc.collect()
