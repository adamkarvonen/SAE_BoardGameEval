from tqdm import tqdm
import pickle
import torch
import einops
from datasets import load_dataset
from typing import Callable, Optional
import math
import os
import itertools
import json
import time

from circuits.utils import (
    get_ae_bundle,
    collect_activations_batch,
    get_model_activations,
    get_feature_activations_batch,
    get_nested_folders,
    get_firing_features,
    to_device,
    AutoEncoderBundle,
)
import circuits.chess_utils as chess_utils
import circuits.othello_utils as othello_utils
import circuits.othello_engine_utils as othello_engine_utils

from circuits.dictionary_learning.evaluation import evaluate

from IPython import embed

# Dimension key (from https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
# F  = features and minibatch size depending on the context (maybe this is stupid)
# B = batch_size
# L = seq length (context length)
# T = thresholds
# R = rows (or cols)
# C = classes for one hot encoding
#
# (rangell): feel free to change these if it doesn't make sense or fall in line with the spirit of the key
# D = activation dimension
# A = all (as opposed to batch B)


def print_tensor_memory_usage(tensor):
    element_size = tensor.element_size()  # size in bytes for one tensor element
    num_elements = tensor.numel()  # number of elements in the tensor
    total_memory = element_size * num_elements  # total memory in bytes
    total_memory /= 1024**2  # total memory in MiB
    print(f"Element size: {element_size} bytes")
    print(f"Number of elements: {num_elements}")
    print(f"Memory usage: {total_memory} MB")


def get_indexing_function_name(indexing_function: Optional[Callable]) -> str:
    if indexing_function is None:
        return "None"
    return indexing_function.__name__


# TODO: Make device consistently use torch.device type hint
def construct_chess_dataset(
    custom_functions: list[Callable],
    n_inputs: int,
    split: str,
    models_path: str = "models/",
    max_str_length: int = 256,
    device: str = "cpu",
    precompute_dataset: bool = True,
) -> dict:
    dataset = load_dataset(
        "adamkarvonen/chess_sae_individual_games_filtered",
        streaming=False,
    )

    meta_path = models_path + "meta.pkl"

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    pgn_strings = []
    encoded_inputs = []
    for i, example in enumerate(dataset[split]):
        if i >= n_inputs:
            break
        pgn_string = example["text"][:max_str_length]
        pgn_strings.append(pgn_string)
        encoded_input = chess_utils.encode_string(meta, pgn_string)
        encoded_inputs.append(encoded_input)

    data = {}
    data["decoded_inputs"] = pgn_strings
    data["encoded_inputs"] = encoded_inputs

    if not precompute_dataset:
        return data

    state_stack_dict_BLRR = chess_utils.create_state_stacks(
        pgn_strings, custom_functions, device, show_progress=True
    )

    for func_name in state_stack_dict_BLRR:
        config = chess_utils.config_lookup[func_name]
        state_stack_BLRR = state_stack_dict_BLRR[func_name]

        assert state_stack_BLRR.shape[0] == len(pgn_strings)
        assert state_stack_BLRR.shape[1] == max_str_length

        one_hot_BLRRC = chess_utils.state_stack_to_one_hot(config, device, state_stack_BLRR)

        print(func_name)
        print_tensor_memory_usage(one_hot_BLRRC)

        data[func_name] = one_hot_BLRRC

    return data


def construct_othello_dataset(
    custom_functions: list[Callable],
    n_inputs: int,
    split: str,
    max_str_length: int = 59,
    device: str = "cpu",
    precompute_dataset: bool = True,
) -> dict:
    dataset = load_dataset("adamkarvonen/othello_45MB_games", streaming=False)
    encoded_othello_inputs = []
    decoded_othello_inputs = []
    for i, example in enumerate(dataset[split]):
        if i >= n_inputs:
            break
        encoded_input = example["tokens"][:max_str_length]
        decoded_input = othello_engine_utils.to_string(encoded_input)
        encoded_othello_inputs.append(encoded_input)
        decoded_othello_inputs.append(decoded_input)

    data = {}
    data["encoded_inputs"] = encoded_othello_inputs
    data["decoded_inputs"] = decoded_othello_inputs

    if not precompute_dataset:
        return data

    for custom_function in custom_functions:
        print(f"Precomputing {custom_function.__name__}...")
        func_name = custom_function.__name__
        data[func_name] = custom_function(decoded_othello_inputs)

    return data


def initialize_results_dict(
    custom_functions: list[Callable],
    num_thresholds: int,
    alive_features_F: torch.Tensor,
    device: torch.device,
) -> dict:
    """For every function for every threshold for every feature, we keep track of the counts for every element
    in the state stack, along with the activations counts. This is done in parallel to make it fast.
    """
    results = {}

    num_features = len(alive_features_F)

    on_counter_TF = torch.zeros(num_thresholds, num_features).to(device)
    results["on_count"] = on_counter_TF
    results["off_count"] = on_counter_TF.clone()
    results["alive_features"] = alive_features_F

    for custom_function in custom_functions:
        results[custom_function.__name__] = {}
        config = chess_utils.config_lookup[custom_function.__name__]
        num_classes = chess_utils.get_num_classes(config)

        results[custom_function.__name__] = {}
        on_tracker_TFRRC = torch.zeros(
            num_thresholds, num_features, config.num_rows, config.num_cols, num_classes
        ).to(device)
        results[custom_function.__name__]["on"] = on_tracker_TFRRC

        all_tracker_RRC = torch.zeros(config.num_rows, config.num_cols, num_classes).to(device)
        results[custom_function.__name__]["all"] = all_tracker_RRC

    return results


def get_data_batch(
    data: dict[str, torch.Tensor],
    inputs_BL: list[str],
    start: int,
    end: int,
    custom_functions: list[Callable],
    device: torch.device,
    precomputed: bool = True,
    othello: bool = False,
) -> dict:
    batch_data = {}
    if precomputed:
        for func_name in data:
            batch_data[func_name] = data[func_name][start:end]
        return batch_data

    if othello:
        for custom_function in custom_functions:
            batch_data[custom_function.__name__] = custom_function(inputs_BL).to(device)
        return batch_data

    # Else construct it on the fly
    state_stacks_dict_BLRR = chess_utils.create_state_stacks(inputs_BL, custom_functions, device)

    for func_name in state_stacks_dict_BLRR:
        config = chess_utils.config_lookup[func_name]
        state_stacks_BLRR = state_stacks_dict_BLRR[func_name]

        batch_data[func_name] = chess_utils.state_stack_to_one_hot(
            config, device, state_stacks_BLRR
        )

    return batch_data


def aggregate_batch_statistics(
    results: dict,
    custom_functions: list[Callable],
    activations_FBL: torch.Tensor,
    thresholds_TF11: torch.Tensor,
    batch_data: dict[str, torch.Tensor],
    f_start: int,
    f_end: int,
    f_batch_size: int,
    device: torch.device,
) -> dict:
    """For every threshold for every activation for every feature, we check if it's above the threshold.
    If so, for every custom function we add the state stack (board or something like pin state) to the on_tracker.
    If not, we add it to the off_tracker.
    We also keep track of how many activations are above and below the threshold (on_count and off_count, respectively)
    This is done in parallel to make it fast."""
    active_indices_TFBL = activations_FBL > thresholds_TF11

    active_counts_TF = einops.reduce(active_indices_TFBL, "T F B L -> T F", "sum")
    off_counts_TF = einops.reduce(~active_indices_TFBL, "T F B L -> T F", "sum")

    results["on_count"][:, f_start:f_end] += active_counts_TF
    results["off_count"][:, f_start:f_end] += off_counts_TF

    for custom_function in custom_functions:
        on_tracker_TFRRC = results[custom_function.__name__]["on"]

        boards_BLRRC = batch_data[custom_function.__name__]
        boards_TFBLRRC = einops.repeat(
            boards_BLRRC,
            "B L R1 R2 C -> T F B L R1 R2 C",
            F=f_batch_size,
            T=thresholds_TF11.shape[0],
        )

        active_boards_sum_TFRRC = einops.reduce(
            boards_TFBLRRC * active_indices_TFBL[:, :, :, :, None, None, None],
            "T F B L R1 R2 C -> T F R1 R2 C",
            "sum",
        )

        on_tracker_TFRRC[:, f_start:f_end, :, :, :] += active_boards_sum_TFRRC

        results[custom_function.__name__]["on"] = on_tracker_TFRRC

    return results


def update_all_tracker(
    results: dict,
    custom_functions: list[Callable],
    batch_data: dict[str, torch.Tensor],
    device: torch.device,
) -> dict:

    for custom_function in custom_functions:
        boards_BLRRC = batch_data[custom_function.__name__]
        all_boards_sum_RRC = einops.reduce(boards_BLRRC, "B L R1 R2 C -> R1 R2 C", "sum")
        results[custom_function.__name__]["all"] += all_boards_sum_RRC

    return results


def normalize_tracker(
    results: dict, tracker_type: str, custom_functions: list[Callable], device: torch.device
):
    """Normalize the specified tracker (on or off) values by its count using element-wise multiplication."""
    for custom_function in custom_functions:
        counts_TF = results[f"{tracker_type}_count"]

        # Calculate inverse of counts safely
        inverse_counts_TF = torch.zeros_like(counts_TF).to(device)
        non_zero_mask = counts_TF > 0
        inverse_counts_TF[non_zero_mask] = 1 / counts_TF[non_zero_mask]

        tracker_TFRRC = results[custom_function.__name__][tracker_type]

        # Normalize using element-wise multiplication
        normalized_tracker_TFRRC = tracker_TFRRC * inverse_counts_TF[:, :, None, None, None]

        # Store the normalized results
        results[custom_function.__name__][f"{tracker_type}_normalized"] = normalized_tracker_TFRRC

    return results


# TODO Write a unit test for this function
def apply_indexing_function(
    pgn_strings: list[str],
    activations_FBL: torch.Tensor,
    batch_data: dict[str, torch.Tensor],
    device: torch.device,
    indexing_function: Callable,
) -> tuple[torch.Tensor, dict]:
    """I'm using `I` in my shape annotation indicating indices.
    If L (seq_len) == 256, there will be around 20 dots indices."""

    max_indices = 20

    custom_indices = []
    for pgn in pgn_strings:
        dots_indices = indexing_function(pgn)
        custom_indices.append(dots_indices[:max_indices])

    custom_indices_BI = torch.tensor(custom_indices).to(device)
    custom_indices_FBI = einops.repeat(
        custom_indices_BI, "B I -> F B I", F=activations_FBL.shape[0]
    )

    activations_FBI = torch.gather(activations_FBL, 2, custom_indices_FBI)

    for custom_function in batch_data:
        boards_BLRRC = batch_data[custom_function]
        rows = boards_BLRRC.shape[2]
        classes = boards_BLRRC.shape[4]
        custom_indices_BIRRC = einops.repeat(
            custom_indices_BI, "B I -> B I R1 R2 C", R1=rows, R2=rows, C=classes
        )
        boards_BIRRC = torch.gather(boards_BLRRC, 1, custom_indices_BIRRC)
        batch_data[custom_function] = boards_BIRRC

    return activations_FBI, batch_data


def compute_custom_indices(
    pgn_strings: list[str],
    indexing_function: Callable,
    num_active_features: int,
    device: str,
) -> torch.Tensor:
    """TODO(rangell):"""
    max_indices = 20

    custom_indices = []
    for pgn in pgn_strings:
        dots_indices = indexing_function(pgn)
        custom_indices.append(dots_indices[:max_indices])

    custom_indices_AI = torch.tensor(custom_indices).to(device)
    return custom_indices_AI


def filter_data_by_custom_indices(
    activations_FBL: torch.Tensor,
    batch_data: dict[str, torch.Tensor],
    custom_indices_BI: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """TODO(rangell):"""

    custom_indices_FBI = einops.repeat(
        custom_indices_BI, "B I -> F B I", F=activations_FBL.shape[0]
    )
    activations_FBI = torch.gather(activations_FBL, 2, custom_indices_FBI)

    for custom_function in batch_data:
        boards_BLRRC = batch_data[custom_function]
        rows = boards_BLRRC.shape[2]
        classes = boards_BLRRC.shape[4]
        custom_indices_BIRRC = einops.repeat(
            custom_indices_BI, "B I -> B I R1 R2 C", R1=rows, R2=rows, C=classes
        )
        boards_BIRRC = torch.gather(boards_BLRRC, 1, custom_indices_BIRRC)
        batch_data[custom_function] = boards_BIRRC

    return activations_FBI, batch_data


def prep_firing_rate_data(
    autoencoder_path: str,
    batch_size: int,
    model_path: str,
    model_name: str,
    data: dict,
    device: torch.device,
    n_inputs: int,
    othello: bool = False,
) -> tuple[dict, AutoEncoderBundle, list[str], torch.Tensor]:
    """Moves data from the data dictionary into the NNsight activation buffer."""
    for key in data:
        if key == "decoded_inputs" or key == "encoded_inputs":
            continue
        data[key] = data[key].to(device)

    pgn_strings = data["decoded_inputs"]
    encoded_inputs = data["encoded_inputs"]
    del data["decoded_inputs"]
    del data["encoded_inputs"]

    firing_rate_data = iter(encoded_inputs)
    n_ctxs = min(512, n_inputs)

    ae_bundle = get_ae_bundle(
        autoencoder_path, device, firing_rate_data, batch_size, model_path, model_name, n_ctxs
    )
    ae_bundle.ae = ae_bundle.ae.to(device)

    return data, ae_bundle, pgn_strings, encoded_inputs


def get_output_location(
    autoencoder_path: str, n_inputs: int, indexing_function: Optional[Callable]
) -> str:
    indexing_function_name = get_indexing_function_name(indexing_function)
    return f"{autoencoder_path}indexing_{indexing_function_name}_n_inputs_{n_inputs}_results.pkl"


def aggregate_statistics(
    custom_functions: list[Callable],
    autoencoder_path: str,
    n_inputs: int,
    batch_size: int,
    device: torch.device,
    model_path: str,
    model_name: str,
    data: dict,
    indexing_function: Optional[Callable] = None,
    othello: bool = False,
    save_results: bool = True,
    precomputed: bool = True,
) -> dict:
    """For every input, for every feature, call `aggregate_batch_statistics()`.
    As an example of desired behavior, view tests/test_classifier_eval.py.
    precomputed will precompute the entire dataset and model activations and store them in memory.
    Faster, but uses far more VRAM."""

    torch.set_grad_enabled(False)
    feature_batch_size = batch_size
    indexing_function_name = get_indexing_function_name(indexing_function)

    data, ae_bundle, pgn_strings, encoded_inputs = prep_firing_rate_data(
        autoencoder_path, batch_size, model_path, model_name, data, device, n_inputs, othello
    )

    firing_rate_n_inputs = min(int(n_inputs * 0.5), 1000) * ae_bundle.context_length

    torch.manual_seed(0)  # For reproducibility
    alive_features_F, max_activations_F = get_firing_features(
        ae_bundle, firing_rate_n_inputs, batch_size, device
    )
    ae_bundle.buffer = None

    if indexing_function is not None:
        custom_indices_AI = compute_custom_indices(
            pgn_strings, indexing_function, alive_features_F.shape[0], device
        )

    num_features = len(alive_features_F)
    print(
        f"Out of {ae_bundle.dictionary_size} features, on {firing_rate_n_inputs} activations, {num_features} are alive."
    )

    assert len(pgn_strings) >= n_inputs
    assert n_inputs % batch_size == 0

    n_iters = n_inputs // batch_size
    # We round up to ensure we don't ignore the remainder of features
    num_feature_iters = math.ceil(num_features / feature_batch_size)

    thresholds_T = torch.arange(0.0, 1.1, 0.1).to(device)
    thresholds_TF11 = einops.repeat(thresholds_T, "T -> T F 1 1", F=num_features)
    max_activations_1F11 = einops.repeat(max_activations_F, "F -> 1 F 1 1")
    thresholds_TF11 = thresholds_TF11 * max_activations_1F11

    results = initialize_results_dict(custom_functions, len(thresholds_T), alive_features_F, device)

    for i in tqdm(range(n_iters), desc="Aggregating statistics"):
        start = i * batch_size
        end = (i + 1) * batch_size
        pgn_strings_BL = pgn_strings[start:end]
        batch_data = get_data_batch(
            data,
            pgn_strings_BL,
            start,
            end,
            custom_functions,
            device,
            precomputed=precomputed,
            othello=othello,
        )

        encoded_inputs_BL = torch.tensor(encoded_inputs[start:end]).to(device)
        all_activations_FBL, tokens = collect_activations_batch(
            ae_bundle, encoded_inputs_BL, alive_features_F
        )

        if indexing_function is not None:
            custom_indices_BI = custom_indices_AI[start:end]
            all_activations_FBL, batch_data = filter_data_by_custom_indices(
                all_activations_FBL, batch_data, custom_indices_BI, device
            )

        results = update_all_tracker(results, custom_functions, batch_data, device)
        # For thousands of features, this would be many GB of memory. So, we minibatch.
        for feature in range(num_feature_iters):
            f_start = feature * feature_batch_size
            f_end = min((feature + 1) * feature_batch_size, num_features)
            f_batch_size = f_end - f_start

            activations_FBL = all_activations_FBL[
                f_start:f_end
            ]  # NOTE: Now F == feature_batch_size
            # Maybe that's stupid and inconsistent and I should use a new letter for annotations
            # I'll roll with it for now

            results = aggregate_batch_statistics(
                results,
                custom_functions,
                activations_FBL,
                thresholds_TF11[:, f_start:f_end, :, :],
                batch_data,
                f_start,
                f_end,
                f_batch_size,
                device,
            )

    autoencoder_config_path = f"{autoencoder_path}config.json"
    with open(autoencoder_config_path, "r") as f:
        trainer_config = json.load(f)

    hyperparameters = {
        "n_inputs": n_inputs,
        "context_length": ae_bundle.context_length,
        "thresholds": thresholds_TF11,
        "indexing_function": indexing_function_name,
    }
    results["hyperparameters"] = hyperparameters
    results["trainer_class"] = trainer_config["trainer"]["trainer_class"]
    results["sae_class"] = ae_bundle.ae._get_name()

    output_location = get_output_location(autoencoder_path, n_inputs, indexing_function)

    if save_results:
        results = to_device(results, "cpu")
        with open(output_location, "wb") as f:
            pickle.dump(results, f)
        results = to_device(results, device)

    return results


def check_if_autoencoder_is_othello(autoencoder_group_path: str) -> bool:
    folders = get_nested_folders(autoencoder_group_path)

    for folder in folders:
        with open(folder + "config.json", "r") as f:
            config = json.load(f)
        if config["buffer"]["ctx_len"] == 59:
            return True
        elif config["buffer"]["ctx_len"] == 256:
            return False
    raise ValueError("Could not determine if autoencoder is for Othello or Chess.")


def get_model_name(othello: bool) -> str:
    if othello:
        return "Baidicoot/Othello-GPT-Transformer-Lens"
    else:
        return "adamkarvonen/8LayerChessGPT2"


def construct_dataset(
    othello: bool,
    custom_functions: list[Callable],
    n_inputs: int,
    split: str,
    device: str,
    models_path: str = "models/",
    precompute_dataset: bool = True,
) -> dict:
    """Constructs the dataset for either Othello or Chess.
    precompute_dataset will precompute the entire dataset and model activations and store them in memory.
    Faster, but uses far more VRAM."""
    if not othello:
        data = construct_chess_dataset(
            custom_functions,
            n_inputs,
            split=split,
            device=device,
            models_path=models_path,
            precompute_dataset=precompute_dataset,
        )
    else:
        data = construct_othello_dataset(
            custom_functions,
            n_inputs,
            split=split,
            device=device,
            precompute_dataset=precompute_dataset,
        )

    return data


def get_recommended_custom_functions(othello: bool) -> list[Callable]:
    if not othello:
        custom_functions = [chess_utils.board_to_piece_state, chess_utils.board_to_pin_state]
    else:
        custom_functions = [
            othello_utils.games_batch_to_state_stack_mine_yours_BLRRC,
            othello_utils.games_batch_to_state_stack_mine_yours_blank_mask_BLRRC,
            othello_utils.games_batch_to_valid_moves_BLRRC,
        ]
    return custom_functions


def get_all_chess_functions(othello: bool) -> list[Callable]:
    if othello:
        raise ValueError("This is a chess function")
    custom_functions = [
        chess_utils.board_to_piece_state,
        chess_utils.board_to_piece_masked_blank_state,
        chess_utils.board_to_piece_masked_blank_and_initial_state,
        chess_utils.board_to_piece_color_state,
        chess_utils.board_to_pin_state,
        chess_utils.board_to_threat_state,
        chess_utils.board_to_check_state,
        chess_utils.board_to_legal_moves_state,
        chess_utils.board_to_specific_fork,
        chess_utils.board_to_any_fork,
        chess_utils.board_to_has_castling_rights,
        chess_utils.board_to_has_queenside_castling_rights,
        chess_utils.board_to_has_kingside_castling_rights,
        chess_utils.board_to_has_legal_en_passant,
        chess_utils.board_to_pseudo_legal_moves_state,
        chess_utils.board_to_can_claim_draw,
        chess_utils.board_to_can_check_next,
        chess_utils.board_to_has_bishop_pair,
        chess_utils.board_to_has_mate_threat,
        chess_utils.board_to_can_capture_queen,
        chess_utils.board_to_has_queen,
        chess_utils.board_to_has_connected_rooks,
        chess_utils.board_to_ambiguous_moves,
    ]
    return custom_functions


def get_recommended_indexing_functions(othello: bool) -> list[Callable]:
    if not othello:
        indexing_functions = [chess_utils.find_dots_indices]
    else:
        indexing_functions = [None]
    return indexing_functions


def eval_sae_group(
    autoencoder_group_paths: list[str],
    device: str = "cuda",
    batch_size: int = 10,
    n_inputs: int = 1000,
):
    """Example autoencoder_group_paths = ['autoencoders/othello_layer5_ef4/'].
    At batch_size == 10, it uses around 2GB of VRAM.
    VRAM does not scale with n_inputs, only batch_size.

    Returns a dictionary with autoencoder_group_path as key and a list of output locations as value.
    """
    model_path = "unused"

    # IMPORTANT NOTE: This is hacky (checks config 'ctx_len'), and means all autoencoders in the group must be for othello XOR chess
    othello = check_if_autoencoder_is_othello(autoencoder_group_paths[0])

    indexing_functions = get_recommended_indexing_functions(othello)
    custom_functions = get_recommended_custom_functions(othello)

    param_combinations = list(itertools.product(autoencoder_group_paths, indexing_functions))

    print("Constructing evaluation dataset...")

    model_name = get_model_name(othello)
    data = construct_dataset(othello, custom_functions, n_inputs, split="train", device=device)

    print("Starting evaluation...")

    for autoencoder_group_path, indexing_function in param_combinations:
        print(f"Autoencoder group path: {autoencoder_group_path}")
        indexing_function_name = get_indexing_function_name(indexing_function)

        print(f"Indexing function: {indexing_function_name}")

        folders = get_nested_folders(autoencoder_group_path)

        for autoencoder_path in folders:
            print("Evaluating autoencoder:", autoencoder_path)

            results = aggregate_statistics(
                custom_functions,
                autoencoder_path,
                n_inputs,
                batch_size,
                device,
                model_path,
                model_name,
                data.copy(),
                indexing_function=indexing_function,
                othello=othello,
            )


if __name__ == "__main__":
    # autoencoder_group_paths = ["autoencoders/group1/", "autoencoders/chess_layer_0_subset/"]
    autoencoder_group_paths = ["autoencoders/othello_layer5_ef4/", "autoencoders/othello_layer0/"]
    autoencoder_group_paths = ["autoencoders/chess_layer5_large_sweep/"]
    autoencoder_group_paths = ["autoencoders/othello_layer5_ef4/"]
    autoencoder_group_paths = ["autoencoders/group-2024-05-07/"]

    eval_sae_group(
        autoencoder_group_paths,
        batch_size=100,
        n_inputs=1000,
    )
