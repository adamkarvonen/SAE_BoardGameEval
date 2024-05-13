from tqdm import tqdm
import pickle
import torch
import einops
from typing import Callable, Optional
import math
import os

from circuits.utils import (
    collect_activations_batch,
    get_nested_folders,
    to_device,
)
import circuits.eval_sae_as_classifier as eval_sae
import circuits.chess_utils as chess_utils


def get_all_feature_label_file_names(folder_name: str) -> list[str]:
    """Get all file names with feature_labels.pkl in the given folder."""
    file_names = []
    for file_name in os.listdir(folder_name):
        if "feature_labels.pkl" in file_name:
            file_names.append(file_name)
    return file_names


def initialize_reconstruction_dict(
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
    counter_T = torch.zeros(num_thresholds).to(device)
    results["on_count"] = on_counter_TF
    results["off_count"] = on_counter_TF.clone()
    results["alive_features"] = alive_features_F
    results["active_per_token"] = counter_T.clone()

    for custom_function in custom_functions:
        results[custom_function.__name__] = {
            "num_boards": 0,
            "num_squares": 0,
            "num_true_positive_squares": 0,
            "num_false_positive_squares": 0,
            "classifiers_per_token": counter_T.clone(),
            "classified_per_token": counter_T.clone(),
        }

    return results


def initialized_constructed_boards_dict(
    custom_functions: list[Callable],
    batch_data: dict[str, torch.Tensor],
    threshold_TF11: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    constructed_boards = {}
    num_thresholds = threshold_TF11.shape[0]
    for custom_function in custom_functions:
        boards_BLRRC = batch_data[custom_function.__name__]
        blank_boards_BLRRC = torch.zeros_like(boards_BLRRC)
        blank_boards_TBLRRC = einops.repeat(
            blank_boards_BLRRC, "B L R1 R2 C -> T B L R1 R2 C", T=num_thresholds
        )
        # I think the .clone() is necessary because einops.repeat() is a view, which torch didn't like
        # for in place operations
        constructed_boards[custom_function.__name__] = blank_boards_TBLRRC.clone().to(device)
    return constructed_boards


def aggregate_feature_labels(
    results: dict,
    feature_labels: dict,
    custom_functions: list[Callable],
    activations_FBL: torch.Tensor,
    thresholds_TF11: torch.Tensor,
    f_start: int,
    f_end: int,
    device: torch.device,
) -> tuple[dict, dict[str, torch.Tensor]]:
    active_indices_TFBL = activations_FBL > thresholds_TF11
    active_indices_TFBL111 = einops.repeat(active_indices_TFBL, "T F B L -> T F B L 1 1 1")

    active_counts_TF = einops.reduce(active_indices_TFBL, "T F B L -> T F", "sum")
    off_counts_TF = einops.reduce(~active_indices_TFBL, "T F B L -> T F", "sum")

    T, F, B, L = active_indices_TFBL.shape
    active_indices_per_token_T = einops.reduce(active_indices_TFBL, "T F B L -> T", "sum") / (L)

    results["on_count"][:, f_start:f_end] += active_counts_TF
    results["off_count"][:, f_start:f_end] += off_counts_TF
    results["active_per_token"] += active_indices_per_token_T

    additive_boards = {}

    for custom_function in custom_functions:
        feature_labels_TFRRC = feature_labels[custom_function.__name__][:, f_start:f_end, :, :, :]

        # NOTE: This will be a very sparse tensor if the L0 is reasonable. TODO: Maybe we can use a sparse tensor?
        # We still have to batch over features in case L0 is large
        feature_labels_TFBLRRC = einops.repeat(
            feature_labels_TFRRC, "T F R1 R2 C -> T F B L R1 R2 C", B=B, L=L
        )

        feature_labels_lookup_TFBLRRC = feature_labels_TFBLRRC * active_indices_TFBL111

        active_boards_sum_TBLRRC = einops.reduce(
            feature_labels_lookup_TFBLRRC,
            "T F B L R1 R2 C -> T B L R1 R2 C",
            "sum",
        )

        active_features_TFBL = einops.reduce(
            feature_labels_lookup_TFBLRRC,
            "T F B L R1 R2 C -> T F B L",
            "sum",
        )

        classifiers_per_token_T = (
            einops.reduce((active_features_TFBL > 0).float(), "T F B L -> T", "sum")
            / (L)
            * (F / (f_end - f_start))
        )

        classified_per_token_T = (
            einops.reduce(active_features_TFBL, "T F B L -> T", "sum")
            / (L)
            * (F / (f_end - f_start))
        )  # scale by num_features / feature batch size
        # TODO Am I scaling by the right thing?

        additive_boards[custom_function.__name__] = active_boards_sum_TBLRRC
        results[custom_function.__name__]["classifiers_per_token"] += classifiers_per_token_T
        results[custom_function.__name__]["classified_per_token"] += classified_per_token_T

    return results, additive_boards


def compare_constructed_to_true_boards(
    results: dict,
    custom_functions: list[Callable],
    constructed_boards: dict[str, torch.Tensor],
    batch_data: dict[str, torch.Tensor],
    device: torch.device,
) -> dict:

    for custom_function in custom_functions:
        true_boards_BLRRC = batch_data[custom_function.__name__]
        constructed_boards_TBLRRC = constructed_boards[custom_function.__name__]

        # Make it binary. Any square with multiple counts is now 1.
        constructed_boards_TBLRRC = (constructed_boards_TBLRRC > 0).int()

        true_bords_TBLRRC = einops.repeat(
            true_boards_BLRRC, "B L R1 R2 C -> T B L R1 R2 C", T=constructed_boards_TBLRRC.shape[0]
        )

        true_positive_TBLRRC = (true_bords_TBLRRC == 1) & (constructed_boards_TBLRRC == 1)
        false_positive_TBLRRC = (true_bords_TBLRRC == 0) & (constructed_boards_TBLRRC == 1)

        true_positive_T = einops.reduce(true_positive_TBLRRC, "T B L R1 R2 C -> T", "sum")
        false_positive_T = einops.reduce(false_positive_TBLRRC, "T B L R1 R2 C -> T", "sum")

        batch_size = true_boards_BLRRC.shape[0]
        num_board_states = true_boards_BLRRC.shape[1]
        num_rows = true_boards_BLRRC.shape[2]
        num_boards = batch_size * num_board_states

        results[custom_function.__name__]["num_boards"] += num_boards
        results[custom_function.__name__]["num_squares"] += num_boards * num_rows * num_rows
        results[custom_function.__name__]["num_true_positive_squares"] += true_positive_T
        results[custom_function.__name__]["num_false_positive_squares"] += false_positive_T

    return results


def normalize_results(results: dict, n_inputs: int, custom_functions: list[Callable]) -> dict:
    results["active_per_token"] /= n_inputs
    for custom_function in custom_functions:
        results[custom_function.__name__]["classifiers_per_token"] /= n_inputs
        results[custom_function.__name__]["classified_per_token"] /= n_inputs

    return results


def calculate_F1_scores(results: dict, custom_functions: list[Callable]) -> dict:
    for custom_function in custom_functions:
        num_squares = results[custom_function.__name__]["num_squares"]
        num_true_positive_squares = results[custom_function.__name__]["num_true_positive_squares"]
        num_false_positive_squares = results[custom_function.__name__]["num_false_positive_squares"]

        false_negatives = num_squares - num_true_positive_squares - num_false_positive_squares

        precision = num_true_positive_squares / (
            num_true_positive_squares + num_false_positive_squares
        )
        recall = num_true_positive_squares / (num_true_positive_squares + false_negatives)

        # Calculate F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall)
        results[custom_function.__name__]["precision"] = precision
        results[custom_function.__name__]["recall"] = recall
        results[custom_function.__name__]["f1_score"] = f1_scores
    return results


def print_out_results(results: dict, custom_functions: list[Callable]):
    print(f"active_per_token", results["active_per_token"])
    print(f"Num alive features: {results['alive_features'].shape[0]}")

    for custom_function in custom_functions:
        print(custom_function.__name__)
        for key in results[custom_function.__name__]:
            print(key, results[custom_function.__name__][key])

        best_idx = results[custom_function.__name__]["f1_score"].argmax()
        f1 = results[custom_function.__name__]["f1_score"][best_idx]
        precision = results[custom_function.__name__]["precision"][best_idx]
        recall = results[custom_function.__name__]["recall"][best_idx]
        num_true_positive_squares = results[custom_function.__name__]["num_true_positive_squares"][
            best_idx
        ]
        num_false_positive_squares = results[custom_function.__name__][
            "num_false_positive_squares"
        ][best_idx]

        active_per_token = results["active_per_token"][best_idx]
        classifiers_per_token = results[custom_function.__name__]["classifiers_per_token"][best_idx]
        classified_per_token = results[custom_function.__name__]["classified_per_token"][best_idx]
        percent_active_classifiers = classifiers_per_token / active_per_token

        print(f"\nBest idx: {best_idx}, best F1: {f1}, Precision: {precision}, Recall: {recall}")
        print(
            f"Num true positives: {num_true_positive_squares}, Num false positives: {num_false_positive_squares}"
        )
        print(
            f"Classifiers per token: {classifiers_per_token}, Classified per token: {classified_per_token}"
        )
        print(
            f"Active per token: {active_per_token}, percent active classifiers: {percent_active_classifiers}"
        )


# Contains some duplicated logic from eval_sae_as_classifier.py
def test_board_reconstructions(
    custom_functions: list[Callable],
    autoencoder_path: str,
    feature_labels: dict,
    output_file: str,
    n_inputs: int,
    batch_size: int,
    device: torch.device,
    model_name: str,
    data: dict,
    othello: bool = False,
    print_results: bool = False,
    save_results: bool = True,
) -> dict:

    torch.set_grad_enabled(False)
    feature_batch_size = batch_size

    data, ae_bundle, pgn_strings, encoded_inputs = eval_sae.prep_firing_rate_data(
        autoencoder_path, batch_size, "", model_name, data, device, n_inputs, othello
    )

    ae_bundle.buffer = None

    thresholds_TF11 = feature_labels["thresholds"].to(device)
    alive_features_F = feature_labels["alive_features"].to(device)
    num_features = len(alive_features_F)
    indexing_function = None

    if feature_labels["indexing_function"] in chess_utils.supported_indexing_functions:
        indexing_function = chess_utils.supported_indexing_functions[
            feature_labels["indexing_function"]
        ]

    feature_labels = to_device(feature_labels, device)

    custom_functions = []
    for key in feature_labels:
        if key in chess_utils.config_lookup:
            custom_functions.append(chess_utils.config_lookup[key].custom_board_state_function)

    results = initialize_reconstruction_dict(
        custom_functions, thresholds_TF11.shape[0], alive_features_F, device
    )

    n_iters = n_inputs // batch_size
    # We round up to ensure we don't ignore the remainder of features
    num_feature_iters = math.ceil(num_features / feature_batch_size)

    for i in tqdm(range(n_iters), desc="Aggregating statistics"):
        start = i * batch_size
        end = (i + 1) * batch_size
        pgn_strings_BL = pgn_strings[start:end]
        encoded_inputs_BL = encoded_inputs[start:end]
        encoded_inputs_BL = torch.tensor(encoded_inputs_BL).to(device)

        batch_data = eval_sae.get_data_batch(
            data, pgn_strings_BL, start, end, custom_functions, device
        )

        all_activations_FBL, encoded_token_inputs = collect_activations_batch(
            ae_bundle, encoded_inputs_BL, alive_features_F
        )

        if indexing_function is not None:
            all_activations_FBL, batch_data = eval_sae.apply_indexing_function(
                pgn_strings[start:end], all_activations_FBL, batch_data, device, indexing_function
            )

        constructed_boards = initialized_constructed_boards_dict(
            custom_functions, batch_data, thresholds_TF11, device
        )

        # For thousands of features, this would be many GB of memory. So, we minibatch.
        for feature in range(num_feature_iters):
            f_start = feature * feature_batch_size
            f_end = min((feature + 1) * feature_batch_size, num_features)
            f_batch_size = f_end - f_start

            activations_FBL = all_activations_FBL[
                f_start:f_end
            ]  # NOTE: Now F == feature_batch_size

            results, additive_boards = aggregate_feature_labels(
                results,
                feature_labels,
                custom_functions,
                activations_FBL,
                thresholds_TF11[:, f_start:f_end, :, :],
                f_start,
                f_end,
                device,
            )

            for custom_function in constructed_boards:
                constructed_boards[custom_function] += additive_boards[custom_function]
        results = compare_constructed_to_true_boards(
            results, custom_functions, constructed_boards, batch_data, device
        )

    results["n_inputs"] = n_inputs
    results = normalize_results(results, n_inputs, custom_functions)
    results = calculate_F1_scores(results, custom_functions)

    if print_results:
        print_out_results(results, custom_functions)

    if save_results:
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
    return results


def test_sae_group_board_reconstruction(
    autoencoder_group_paths: list[str],
    device: str = "cuda",
    batch_size: int = 10,
    n_inputs: int = 1000,
    print_results: bool = False,
    save_results: bool = True,
):
    """Example autoencoder_group_paths = ['autoencoders/othello_layer5_ef4/'].
    At batch_size == 10, it uses around 2GB of VRAM.
    VRAM does not scale with n_inputs, only batch_size."""

    torch.set_printoptions(sci_mode=False, precision=2)

    print("Starting evaluation...")

    for autoencoder_group_path in autoencoder_group_paths:
        print(f"Autoencoder group path: {autoencoder_group_path}")

        othello = eval_sae.check_if_autoencoder_is_othello(autoencoder_group_path)
        model_name = eval_sae.get_model_name(othello)

        folders = get_nested_folders(autoencoder_group_path)

        if len(folders) == 0:
            print("No autoencoders found in this folder.")
            continue

        # All of this fiddling around is to make sure we have the right custom functions
        # So we only have to construct the evaluation dataset once
        first_folder = folders[0]

        feature_label_files = get_all_feature_label_file_names(first_folder)

        if len(feature_label_files) == 0:
            print("No feature label files found in this folder.")
            continue

        first_feature_labels = feature_label_files[0]
        with open(first_folder + first_feature_labels, "rb") as f:
            feature_labels = pickle.load(f)

        custom_functions = []
        for key in feature_labels:
            if key in chess_utils.config_lookup:
                custom_functions.append(chess_utils.config_lookup[key].custom_board_state_function)

        print("Constructing evaluation dataset...")

        # TODO: This is pretty hacky. It assumes that all autoencoder_group_paths are othello XOR chess
        # It shouldn't be too hard to make it smarter
        data = eval_sae.construct_dataset(othello, custom_functions, n_inputs, device)

        for autoencoder_path in folders:

            print("\n\n\nTesting autoencoder:", autoencoder_path)
            feature_label_files = get_all_feature_label_file_names(autoencoder_path)

            for feature_label_file in feature_label_files:
                print("Testing feature label file:", feature_label_file)
                output_file = feature_label_file.replace("feature_labels.pkl", "reconstruction.pkl")

                with open(autoencoder_path + feature_label_file, "rb") as f:
                    feature_labels = pickle.load(f)

                test_board_reconstructions(
                    custom_functions,
                    autoencoder_path,
                    feature_labels,
                    output_file,
                    n_inputs,
                    batch_size,
                    device,
                    model_name,
                    data.copy(),
                    othello=othello,
                    print_results=print_results,
                    save_results=save_results,
                )


if __name__ == "__main__":
    autoencoder_group_paths = ["autoencoders/othello_layer5_ef4/", "autoencoders/othello_layer0/"]
    autoencoder_group_paths = ["autoencoders/chess_layer5_large_sweep/"]
    autoencoder_group_paths = ["autoencoders/othello_layer5_ef4/"]

    test_sae_group_board_reconstruction(autoencoder_group_paths)
