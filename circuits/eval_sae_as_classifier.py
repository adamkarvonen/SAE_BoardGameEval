from tqdm import tqdm
import pickle
import torch
import einops
from datasets import load_dataset

from circuits.utils import get_ae_bundle, collect_activations_batch, get_nested_folders
import circuits.chess_utils as chess_utils

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
    print(f"Memory usage: {total_memory} bytes")


# TODO Is there a better type hint for callable?
def construct_eval_dataset(
    custom_functions: list[callable],
    n_inputs: int,
    output_path: str = "data.pkl",
    max_str_length: int = 256,
    device: str = "cpu",
):
    dataset = load_dataset("adamkarvonen/chess_sae_individual_games_filtered", streaming=False)
    pgn_strings = []
    for i, example in enumerate(dataset["train"]):
        if i >= n_inputs:
            break
        pgn_strings.append(example["text"][:max_str_length])

    data = {}
    data["pgn_strings"] = pgn_strings

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


def initialize_results_dict(
    custom_functions: list[callable], num_thresholds: int, num_features: int, device: torch.device
) -> dict:
    """For every function for every threshold for every feature, we keep track of the counts for every element
    in the state stack, along with the activations counts. This is done in parallel to make it fast.
    """
    results = {}

    on_counter_TF = torch.zeros(num_thresholds, num_features).to(device)
    results["on_count"] = on_counter_TF
    results["off_count"] = on_counter_TF.clone()

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
    custom_functions: list[callable],
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
        if config.num_rows == 8:
            state_stacks = chess_utils.create_state_stacks(inputs_BL, custom_function).to(device)
            batch_data[custom_function.__name__] = chess_utils.state_stack_to_one_hot(
                config, device, state_stacks
            )
        else:
            batch_data[custom_function.__name__] = data[custom_function.__name__][start:end]

    return batch_data


def aggregate_batch_statistics(
    results: dict,
    custom_functions: list[callable],
    activations_FBL: torch.Tensor,
    thresholds_T111: torch.Tensor,
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
    active_indices_TFBL = activations_FBL > thresholds_T111

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
            T=len(thresholds_T111),
        )

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
    results: dict, tracker_type: str, custom_functions: list[callable], device: torch.device
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


def aggregate_statistics(
    custom_functions: list[callable],
    autoencoder_path: str,
    n_inputs: int,
    batch_size: int,
    device: torch.device,
    model_path: str,
    data_path: str,
):
    """For every input, for every feature, call `aggregate_batch_statistics()`.
    As an example of desired behavior, view tests/test_classifier_eval.py."""

    torch.set_grad_enabled(False)
    feature_batch_size = batch_size

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    for key in data:
        if key == "pgn_strings":
            continue
        data[key] = data[key].to(device)

    pgn_strings = data["pgn_strings"]
    del data["pgn_strings"]

    ae_bundle = get_ae_bundle(autoencoder_path, device, data, batch_size, model_path)
    ae_bundle.buffer = None

    features = torch.arange(0, ae_bundle.dictionary_size, device=device)
    num_features = len(features)

    assert len(pgn_strings) >= n_inputs
    assert n_inputs % batch_size == 0

    n_iters = n_inputs // batch_size
    num_feature_iters = num_features // feature_batch_size

    thresholds_T111 = (
        torch.arange(0.0, 1.0, 0.1).view(-1, 1, 1, 1).to(device)
    )  # Reshape for broadcasting

    results = initialize_results_dict(custom_functions, len(thresholds_T111), num_features, device)

    for i in tqdm(range(n_iters)):
        start = i * batch_size
        end = (i + 1) * batch_size
        inputs_BL = pgn_strings[start:end]

        batch_data = get_data_batch(data, inputs_BL, start, end, custom_functions, device)

        all_activations_FBL, encoded_inputs = collect_activations_batch(
            ae_bundle, inputs_BL, features
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
                thresholds_T111,
                batch_data,
                f_start,
                f_end,
                f_batch_size,
                device,
            )

    autoencoder_results_name = autoencoder_path.replace("/", "_") + "_results.pkl"
    with open(autoencoder_results_name, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    custom_functions = [chess_utils.board_to_piece_state, chess_utils.board_to_pin_state]

    autoencoder_group_path = "autoencoders/group0/"

    folders = get_nested_folders(autoencoder_group_path)

    batch_size = 10
    feature_batch_size = 10
    n_inputs = 100
    device = "cuda"
    # device = "cpu"
    model_path = "models/"
    data_path = "data.pkl"

    construct_eval_dataset(custom_functions, n_inputs, output_path=data_path, device="cpu")

    for autoencoder_path in folders:
        aggregate_statistics(
            custom_functions, autoencoder_path, n_inputs, batch_size, device, model_path, data_path
        )
