from tqdm import tqdm
import pickle
import torch
import einops
from datasets import load_dataset
from typing import Callable, Optional
import math

from circuits.utils import (
    get_ae_bundle,
    collect_activations_batch,
    get_nested_folders,
    get_firing_features,
    to_device,
    AutoEncoderBundle,
)
import circuits.chess_utils as chess_utils
import circuits.othello_utils as othello_utils
import circuits.othello_engine_utils as othello_engine_utils

# Dimension key (from https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
# F  = features and minibatch size depending on the context (maybe this is stupid)
# B = batch_size
# L = seq length (context length)
# T = thresholds
# R = rows (or cols)
# C = classes for one hot encoding


def print_tensor_memory_usage(tensor):
    element_size = tensor.element_size()  # size in bytes for one tensor element
    num_elements = tensor.numel()  # number of elements in the tensor
    total_memory = element_size * num_elements  # total memory in bytes
    total_memory /= 1024**2  # total memory in MiB
    print(f"Element size: {element_size} bytes")
    print(f"Number of elements: {num_elements}")
    print(f"Memory usage: {total_memory} MB")


# TODO: Make device consistently use torch.device type hint
def construct_eval_dataset(
    custom_functions: list[Callable],
    n_inputs: int,
    models_path: str = "models/",
    output_path: str = "data.pkl",
    max_str_length: int = 256,
    device: str = "cpu",
):
    dataset = load_dataset("adamkarvonen/chess_sae_individual_games_filtered", streaming=False)

    meta_path = models_path + "meta.pkl"

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    pgn_strings = []
    encoded_inputs = []
    for i, example in enumerate(dataset["train"]):
        if i >= n_inputs:
            break
        pgn_string = example["text"][:max_str_length]
        pgn_strings.append(pgn_string)
        encoded_input = chess_utils.encode_string(meta, pgn_string)
        encoded_inputs.append(encoded_input)

    data = {}
    data["decoded_inputs"] = pgn_strings
    data["encoded_inputs"] = encoded_inputs

    for function in custom_functions:
        func_name = function.__name__
        config = chess_utils.config_lookup[func_name]
        if config.num_rows == 8:
            continue
        func_name = config.custom_board_state_function.__name__
        state_stack_BLRR = chess_utils.create_state_stacks(
            pgn_strings, config.custom_board_state_function
        ).to(device)

        assert state_stack_BLRR.shape[0] == len(pgn_strings)
        assert state_stack_BLRR.shape[1] == max_str_length

        one_hot_BLRRC = chess_utils.state_stack_to_one_hot(config, device, state_stack_BLRR)

        print(func_name)
        print_tensor_memory_usage(one_hot_BLRRC)

        data[func_name] = one_hot_BLRRC

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    return data


def construct_othello_dataset(
    custom_functions: list[Callable],
    n_inputs: int,
    output_path: str = "data.pkl",
    max_str_length: int = 59,
    device: str = "cpu",
):
    """Because we are dealing with 8x8 state stacks, I won't bother creating any state stacks in advance."""
    dataset = load_dataset("adamkarvonen/othello_45MB_games", streaming=False)
    encoded_othello_inputs = []
    decoded_othello_inputs = []
    for i, example in enumerate(dataset["train"]):
        if i >= n_inputs:
            break
        encoded_input = example["tokens"][:max_str_length]
        decoded_input = othello_engine_utils.to_string(encoded_input)
        encoded_othello_inputs.append(encoded_input)
        decoded_othello_inputs.append(decoded_input)

    data = {}
    data["encoded_inputs"] = encoded_othello_inputs
    data["decoded_inputs"] = decoded_othello_inputs

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

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
        results[custom_function.__name__]["off"] = on_tracker_TFRRC.clone()

    return results


def get_data_batch(
    data: dict[str, torch.Tensor],
    inputs_BL: list[str],
    start: int,
    end: int,
    custom_functions: list[Callable],
    device: torch.device,
) -> dict:
    """If the custom function returns a board of 8 x 8 x num_classes, we construct it on the fly.
    In this case, creating the state stack is very cheap compared to doing the statistics aggregation.
    Additionally, a full board state stack very quickly grows to dozens of gigabytes, so we don't want to store it.

    However, if the custom function returns a 1 x 1 x num_classes tensor, creating the state stack is comparable to the statistics aggregation.
    And memory usage is low, so it makes sense to compute the state stack once and store it."""
    batch_data = {}
    for custom_function in custom_functions:
        config = chess_utils.config_lookup[custom_function.__name__]
        if custom_function.__name__ in othello_utils.othello_functions:
            batch_data[custom_function.__name__] = custom_function(inputs_BL).to(device)
        else:
            if config.num_rows == 8:
                state_stacks = chess_utils.create_state_stacks(inputs_BL, custom_function).to(
                    device
                )
                batch_data[custom_function.__name__] = chess_utils.state_stack_to_one_hot(
                    config, device, state_stacks
                )
            else:
                batch_data[custom_function.__name__] = data[custom_function.__name__][start:end]
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
        off_tracker_FTRRC = results[custom_function.__name__]["off"]

        boards_BLRRC = batch_data[custom_function.__name__]
        boards_TFBLRRC = einops.repeat(
            boards_BLRRC,
            "B L R1 R2 C -> T F B L R1 R2 C",
            F=f_batch_size,
            T=thresholds_TF11.shape[0],
        )

        # TODO The next 2 operations consume almost all of the compute. I don't think it will work,
        # but maybe we can only do 1 of these operations?
        active_boards_sum_TFRRC = einops.reduce(
            boards_TFBLRRC * active_indices_TFBL[:, :, :, :, None, None, None],
            "T F B L R1 R2 C -> T F R1 R2 C",
            "sum",
        )
        off_boards_sum_TFRRC = einops.reduce(
            boards_TFBLRRC * ~active_indices_TFBL[:, :, :, :, None, None, None],
            "T F B L R1 R2 C -> T F R1 R2 C",
            "sum",
        )

        on_tracker_TFRRC[:, f_start:f_end, :, :, :] += active_boards_sum_TFRRC
        off_tracker_FTRRC[:, f_start:f_end, :, :, :] += off_boards_sum_TFRRC

        results[custom_function.__name__]["on"] = on_tracker_TFRRC
        results[custom_function.__name__]["off"] = off_tracker_FTRRC

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
    activations_FBL,
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


def prep_firing_rate_data(
    data: dict, device: torch.device, model_name: str, othello: bool = False
) -> tuple[dict, AutoEncoderBundle, list[str], torch.Tensor]:
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

    return data, ae_bundle, pgn_strings, encoded_inputs


def aggregate_statistics(
    custom_functions: list[Callable],
    autoencoder_path: str,
    n_inputs: int,
    batch_size: int,
    device: torch.device,
    model_path: str,
    model_name: str,
    data_path: str,
    indexing_function: Optional[Callable] = None,
    othello: bool = False,
):
    """For every input, for every feature, call `aggregate_batch_statistics()`.
    As an example of desired behavior, view tests/test_classifier_eval.py."""

    torch.set_grad_enabled(False)
    feature_batch_size = batch_size

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    data, ae_bundle, pgn_strings, encoded_inputs = prep_firing_rate_data(
        data, device, model_name, othello
    )

    firing_rate_n_inputs = min(int(n_inputs * 0.5), 1000) * ae_bundle.context_length
    # TODO: Custom thresholds per feature based on max activations
    alive_features_F, max_activations_F = get_firing_features(
        ae_bundle, firing_rate_n_inputs, batch_size, device
    )
    ae_bundle.buffer = None
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
        encoded_inputs_BL = encoded_inputs[start:end]
        encoded_inputs_BL = torch.tensor(encoded_inputs_BL).to(device)

        batch_data = get_data_batch(data, pgn_strings_BL, start, end, custom_functions, device)

        all_activations_FBL, encoded_token_inputs = collect_activations_batch(
            ae_bundle, encoded_inputs_BL, alive_features_F
        )

        if indexing_function is not None:
            all_activations_FBL, batch_data = apply_indexing_function(
                pgn_strings[start:end], all_activations_FBL, batch_data, device, indexing_function
            )
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

    hyperparameters = {
        "n_inputs": n_inputs,
        "context_length": ae_bundle.context_length,
        "thresholds": thresholds_TF11,
        "indexing_function": None,
    }
    if indexing_function is not None:
        hyperparameters["indexing_function"] = indexing_function.__name__
    results["hyperparameters"] = hyperparameters

    results = to_device(results, "cpu")
    autoencoder_results_name = autoencoder_path.replace("/", "_") + "results.pkl"
    with open(autoencoder_results_name, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":

    othello = True

    # At these settings, it uses around 3GB of VRAM
    # VRAM does not scale with n_inputs, only batch_size
    # You can increase batch_size if you have more VRAM, but it's not a large speedup
    batch_size = 10
    feature_batch_size = 10
    n_inputs = 1000
    device = "cuda"
    # device = "cpu"
    model_path = "models/"
    data_path = "data.pkl"

    print("Constructing evaluation dataset...")

    if not othello:
        autoencoder_group_path = "autoencoders/group1/"
        model_name = "adamkarvonen/8LayerChessGPT2"
        custom_functions = [chess_utils.board_to_piece_state, chess_utils.board_to_pin_state]
        construct_eval_dataset(custom_functions, n_inputs, output_path=data_path, device="cpu")

    else:
        autoencoder_group_path = "autoencoders/othello_layer5_ef4/"
        model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
        custom_functions = [othello_utils.games_batch_to_state_stack_BLRRC]
        construct_othello_dataset(custom_functions, n_inputs, output_path=data_path, device="cpu")

    print("Starting evaluation...")

    folders = get_nested_folders(autoencoder_group_path)
    for autoencoder_path in folders:
        print("Evaluating autoencoder:", autoencoder_path)
        aggregate_statistics(
            custom_functions,
            autoencoder_path,
            n_inputs,
            batch_size,
            device,
            model_path,
            model_name,
            data_path,
            indexing_function=None,
            othello=othello,
        )
