import argparse
import torch as t
import numpy as np
import itertools
import pickle

from nnsight import LanguageModel

#from circuits.nanogpt_to_hf_transformers import NanogptTokenizer, convert_nanogpt_model
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
from dictionary_learning.trainers.top_k import AutoEncoderTopK, TrainerTopK
from dictionary_learning.utils import hf_dataset_to_generator, zst_to_generator
from dictionary_learning.buffer import ActivationBuffer, NNsightActivationBuffer
from dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpAutoEncoder,
)

from joblib import Parallel, delayed

from IPython import embed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, choices=["chess", "othello"],
                        required=True, help="data and model and context length to run on")
    parser.add_argument("--layer", type=int, required=True,
                        help="which residual stream layer to gather activations from")
    parser.add_argument("--trainer_type", type=str, choices=["standard", "p_anneal", "gated", "gated_anneal", "top_k"],
                        required=True, help="run sweep on this trainer")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="where to store sweep")
    parser.add_argument("--random_model", action="store_true",  help="use random weight LM")
    parser.add_argument("--dry_run", action="store_true",  help="dry run sweep")
    args = parser.parse_args()
    return args


def run_sae_batch(
        othello : bool,
        layer : int,
        trainer_type : str,
        save_dir : str,
        device : str,
        random_model : bool = False,
        dry_run : bool = False
):
    if not othello:
        with open("circuits/resources/meta.pkl", "rb") as f:
            meta = pickle.load(f)

        context_length = 256
        if random_model:
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
        if random_model:
            model_name = "adamkarvonen/RandomWeights8LayerOthelloGPT2"
        else:
            model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
        dataset_name = "taufeeque/othellogpt"
        data = othello_hf_dataset_to_generator(
            dataset_name, context_length=context_length, split="train", streaming=True
        )
        model_type = "othello"

    model = get_model(model_name, device)
    submodule = get_submodule(model_name, layer, model)

    activation_dim = 512  # output dimension of the layer

    buffer_size = int(1e4 / 1)
    llm_batch_size = 256 # 256 for A100 GPU, 64 for 1080ti
    sae_batch_size = 8192

    num_tokens = 300_000_000

    activation_buffer = NNsightActivationBuffer(
        data,
        model,
        submodule,
        n_ctxs=1e3,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io="out",
        d_submodule=activation_dim,
        device=device,
    )

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

    learning_rate_ = [3e-4]
    expansion_factor_ = (2 ** t.arange(3, 5)).tolist()
    sparsity_queue_length_ = [10]
    anneal_start_ = [10000]
    n_sparsity_updates_ = [10]
    if trainer_type == "p_anneal":
        #initial_sparsity_penalty_ = t.linspace(0.02, 0.08, 20).tolist()    # chess
        initial_sparsity_penalty_ = t.linspace(0.025, 0.05, 20).tolist()    # othello
        param_combinations = itertools.product(
            learning_rate_,
            expansion_factor_,
            sparsity_queue_length_,
            anneal_start_,
            n_sparsity_updates_,
            initial_sparsity_penalty_,
        )

        print(f"Sweep parameters for {trainer_type}: ")
        print("learning_rate: ", [round(x, 4) for x in learning_rate_])
        print("expansion_factor: ", [round(x, 4) for x in expansion_factor_])
        print("sparsity_queue_length: ", [round(x, 4) for x in sparsity_queue_length_])
        print("anneal_start: ", [round(x, 4) for x in anneal_start_])
        print("n_sparsity_updates: ", [round(x, 4) for x in n_sparsity_updates_])
        print("initial_sparsity_penalty: ", [round(x, 4) for x in initial_sparsity_penalty_])

        for i, param_setting in enumerate(param_combinations):
            lr, expansion_factor, sparsity_queue_length, anneal_start, n_sparsity_updates, sp = (
                param_setting
            )

            trainer_configs.append(
                {
                    "trainer": PAnnealTrainer,
                    "dict_class": AutoEncoder,
                    "activation_dim": activation_dim,
                    "dict_size": expansion_factor * activation_dim,
                    "lr": lr,
                    "sparsity_function": "Lp^p",
                    "initial_sparsity_penalty": sp,
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
                    "wandb_name": f"PAnnealTrainer-{model_type}-{i}",
                    "layer" : layer,
                    "lm_name" : model_name,
                    "device": device,
                }
            )
    elif trainer_type == "standard":
        #initial_sparsity_penalty_ = t.linspace(0.03, 0.1, 20).tolist()    # chess
        initial_sparsity_penalty_ = t.linspace(0.035, 0.08, 20).tolist()    # othello
        param_combinations = itertools.product(
            learning_rate_, expansion_factor_, initial_sparsity_penalty_
        )

        print(f"Sweep parameters for {trainer_type}: ")
        print("learning_rate: ", [round(x, 4) for x in learning_rate_])
        print("expansion_factor: ", [round(x, 4) for x in expansion_factor_])
        print("initial_sparsity_penalty: ", [round(x, 4) for x in initial_sparsity_penalty_])

        for i, param_setting in enumerate(param_combinations):
            lr, expansion_factor, sp = param_setting
            trainer_configs.append(
                {
                    "trainer": StandardTrainer,
                    "dict_class": AutoEncoder,
                    "activation_dim": activation_dim,
                    "dict_size": expansion_factor * activation_dim,
                    "lr": lr,
                    "l1_penalty": sp,
                    "warmup_steps": warmup_steps,
                    "resample_steps": resample_steps,
                    "seed": seed,
                    "layer" : layer,
                    "lm_name" : model_name,
                    "wandb_name": f"StandardTrainer-{model_type}-{i}",
                    "device": device,
                }
            )
    elif trainer_type == "gated":
        #initial_sparsity_penalty_ = t.linspace(0.15, 1.0, 20).tolist()   # chess
        initial_sparsity_penalty_ = t.linspace(1.0, 2.0, 20).tolist()   # othello
        param_combinations = itertools.product(
            learning_rate_, expansion_factor_, initial_sparsity_penalty_
        )

        print(f"Sweep parameters for {trainer_type}: ")
        print("learning_rate: ", [round(x, 4) for x in learning_rate_])
        print("expansion_factor: ", [round(x, 4) for x in expansion_factor_])
        print("initial_sparsity_penalty: ", [round(x, 4) for x in initial_sparsity_penalty_])

        for i, param_setting in enumerate(param_combinations):
            lr, expansion_factor, sp = param_setting
            trainer_configs.append(
                {
                    "trainer": GatedSAETrainer,
                    "dict_class": GatedAutoEncoder,
                    "activation_dim": activation_dim,
                    "dict_size": expansion_factor * activation_dim,
                    "lr": lr,
                    "l1_penalty": sp,
                    "warmup_steps": warmup_steps,
                    "resample_steps": resample_steps,
                    "seed": seed,
                    "layer" : layer,
                    "lm_name" : model_name,
                    "wandb_name": f"GatedSAETrainer-{model_type}-{i}",
                    "device": device,
                }
            )
    elif trainer_type == "gated_anneal":
        #initial_sparsity_penalty_ = t.linspace(0.05, 0.15, 20).tolist()   # chess
        initial_sparsity_penalty_ = t.linspace(0.15, 1.0, 20).tolist()    # othello
        param_combinations = itertools.product(
            learning_rate_,
            expansion_factor_,
            sparsity_queue_length_,
            anneal_start_,
            n_sparsity_updates_,
            initial_sparsity_penalty_,
        )

        print(f"Sweep parameters for {trainer_type}: ")
        print("learning_rate: ", [round(x, 4) for x in learning_rate_])
        print("expansion_factor: ", [round(x, 4) for x in expansion_factor_])
        print("sparsity_queue_length: ", [round(x, 4) for x in sparsity_queue_length_])
        print("anneal_start: ", [round(x, 4) for x in anneal_start_])
        print("n_sparsity_updates: ", [round(x, 4) for x in n_sparsity_updates_])
        print("initial_sparsity_penalty: ", [round(x, 4) for x in initial_sparsity_penalty_])

        for i, param_setting in enumerate(param_combinations):
            lr, expansion_factor, sparsity_queue_length, anneal_start, n_sparsity_updates, sp = (
                param_setting
            )

            trainer_configs.append(
                {
                    "trainer": GatedAnnealTrainer,
                    "dict_class": GatedAutoEncoder,
                    "activation_dim": activation_dim,
                    "dict_size": expansion_factor * activation_dim,
                    "lr": lr,
                    "sparsity_function": "Lp^p",
                    "initial_sparsity_penalty": sp,
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
                    "layer" : layer,
                    "lm_name" : model_name,
                    "wandb_name": f"GatedAnnealTrainer-{model_type}-{i}",
                    "device": device,
                }
            )
    elif trainer_type == "top_k":
        k_ = [int(k) for k in t.linspace(10, 400, 40)]
        param_combinations = itertools.product(
            expansion_factor_, k_
        )

        print(f"Sweep parameters for {trainer_type}: ")
        print("k: ", k_)

        for i, param_setting in enumerate(param_combinations):
            expansion_factor, k = param_setting
            trainer_configs.append(
                {
                    "trainer": TrainerTopK,
                    "dict_class": AutoEncoderTopK,
                    "activation_dim": activation_dim,
                    "dict_size": expansion_factor * activation_dim,
                    "k" : k,
                    "steps": steps,
                    "seed": seed,
                    "layer" : layer,
                    "lm_name" : model_name,
                    "wandb_name": f"TrainerTopK-{model_type}-{i}",
                    "device": device,
                }
            )
    else:
        raise ValueError("Unknown trainer type: ", trainer_type)

    print(f"len trainer configs: {len(trainer_configs)}")

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
    run_sae_batch(
        args.game == "othello",
        args.layer,
        args.trainer_type,
        args.save_dir,
        "cuda:0",
        random_model=args.random_model,
        dry_run=args.dry_run,
    )
