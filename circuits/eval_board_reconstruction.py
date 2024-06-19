from tqdm import tqdm
import pickle
import torch
import einops
from typing import Callable, Optional
import math
import os

from circuits.utils import (
    collect_activations_batch,
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
            "num_true_negative_squares": 0,
            "num_false_negative_squares": 0,
            "num_multiple_classes": 0,
            "num_true_and_false_positive_squares": 0,
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


# def create_max_active_mask(
#     active_indices_TFBL: torch.Tensor,  # feature_labels_TFRRC: torch.Tensor
# ) -> torch.Tensor:

#     total_nonzero_acts = active_indices_TFBL[0].sum()
#     T, f, B, L = active_indices_TFBL.shape

#     active_indices_FBL = einops.reduce(active_indices_TFBL, "T F B L -> F B L", "sum") - 1
#     zeros_indices_FBL = active_indices_FBL == -1
#     zeros_indices_TFBL = einops.repeat(zeros_indices_FBL, "F B L -> T F B L", T=T)
#     active_indices_FBL = active_indices_FBL.clamp(min=0)

#     max_indices_FBLT = F.one_hot(active_indices_FBL, num_classes=T)
#     max_indices_TFBL = einops.rearrange(max_indices_FBLT, "F B L T -> T F B L")
#     max_indices_TFBL = max_indices_TFBL * ~zeros_indices_TFBL

#     max_acts = max_indices_TFBL.sum()

#     assert max_acts == total_nonzero_acts, f"max_acts: {max_acts}, total_acts: {total_nonzero_acts}"

#     # print(
#     #     f"Total activations: {total_nonzero_acts}, Max activations: {max_acts}, Total squares: {total_acts}, Max squares: {nonzero_max_acts}"
#     # )

#     return max_indices_TFBL


def create_max_active_mask(
    active_indices_TFBL: torch.Tensor, feature_labels_TFRRC: torch.Tensor
) -> torch.Tensor:
    # Calculate the total activations for assertion later
    total_nonzero_acts = active_indices_TFBL[0].sum()

    # Generate a range tensor that increments along the T dimension
    T = active_indices_TFBL.shape[0]
    range_tensor_T111 = torch.arange(T, device=active_indices_TFBL.device)[:, None, None, None]

    # Multiply active indices by the range to mark the indices with their actual time step values
    weighted_active_indices_TFBL = active_indices_TFBL.float() * range_tensor_T111

    feature_square_counts_TF = einops.reduce(feature_labels_TFRRC, "T F R1 R2 C -> T F", "sum")
    # Broadcast feature_square_counts_TF to match the shape of weighted_active_indices_TFBL
    feature_square_counts_TF11 = einops.repeat(feature_square_counts_TF, "T F -> T F 1 1")

    # Mask out the weighted indices where the corresponding feature count is zero
    mask_TFBL = feature_square_counts_TF11 > 0
    weighted_active_indices_TFBL = weighted_active_indices_TFBL * mask_TFBL

    # Use argmax to find the last time step index that was active (last occurrence of 1)
    max_indices_FBL = torch.argmax(weighted_active_indices_TFBL, dim=0)

    # Convert to one-hot encoding across the T dimension
    max_indices_FBLT = torch.nn.functional.one_hot(max_indices_FBL, num_classes=T)
    max_indices_TFBL = einops.rearrange(max_indices_FBLT, "F B L T -> T F B L")

    # Ensure that only the places with at least one activation across T are considered
    valid_activations_FBL = einops.reduce(active_indices_TFBL, "T F B L -> F B L", "sum") > 0
    max_indices_TFBL = max_indices_TFBL * einops.repeat(
        valid_activations_FBL, "F B L -> T F B L", T=T
    )

    # Ensure the sum of the resulting mask is equal to the total number of nonzero activations
    assert max_indices_TFBL.sum() == total_nonzero_acts, "Mismatch in activation counts!"

    return max_indices_TFBL


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

        active_boards_sum_TBLRRC = einops.reduce(
            feature_labels_TFBLRRC * active_indices_TFBL111,
            "T F B L R1 R2 C -> T B L R1 R2 C",
            "sum",
        )

        max_indices_TFBL = create_max_active_mask(active_indices_TFBL, feature_labels_TFRRC)
        max_indices_TFBL111 = einops.repeat(max_indices_TFBL, "T F B L -> T F B L 1 1 1")

        max_boards_sum_BLRRC = einops.reduce(
            feature_labels_TFBLRRC * max_indices_TFBL111,
            "T F B L R1 R2 C -> B L R1 R2 C",
            "sum",
        )

        active_boards_sum_TBLRRC[-1] = max_boards_sum_BLRRC

        additive_boards[custom_function.__name__] = active_boards_sum_TBLRRC

        # The following code would be useful to look at the number of classifiers and classified per token on a per turn basis
        # As it currently is, analysis.py could give similar results in much less time
        # active_features_TFBL = einops.reduce(
        #     feature_labels_lookup_TFBLRRC,
        #     "T F B L R1 R2 C -> T F B L",
        #     "sum",
        # )

        # classifiers_per_token_T = (
        #     einops.reduce((active_features_TFBL > 0).float(), "T F B L -> T", "sum")
        #     / (L)
        #     * (F / (f_end - f_start))
        # )

        # classified_per_token_T = (
        #     einops.reduce(active_features_TFBL, "T F B L -> T", "sum")
        #     / (L)
        #     * (F / (f_end - f_start))
        # )  # scale by num_features / feature batch size
        # # TODO Am I scaling by the right thing?

        # results[custom_function.__name__]["classifiers_per_token"] += classifiers_per_token_T
        # results[custom_function.__name__]["classified_per_token"] += classified_per_token_T

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

        multiple_classes_TBLRR = einops.reduce(
            constructed_boards_TBLRRC, "T B L R1 R2 C -> T B L R1 R2", "sum"
        )
        multiple_classes_TBLRR = (multiple_classes_TBLRR > 1).int()
        multiple_classes_T = einops.reduce(multiple_classes_TBLRR, "T B L R1 R2 -> T", "sum")

        true_positive_TBLRRC = (true_bords_TBLRRC == 1) & (constructed_boards_TBLRRC == 1)
        false_positive_TBLRRC = (true_bords_TBLRRC == 0) & (constructed_boards_TBLRRC == 1)
        true_negative_TBLRRC = (true_bords_TBLRRC == 0) & (constructed_boards_TBLRRC == 0)
        false_negative_TBLRRC = (true_bords_TBLRRC == 1) & (constructed_boards_TBLRRC == 0)

        true_positive_TBLRR = einops.reduce(
            true_positive_TBLRRC, "T B L R1 R2 C -> T B L R1 R2", "sum"
        )
        false_positive_TBLRR = einops.reduce(
            false_positive_TBLRRC, "T B L R1 R2 C -> T B L R1 R2", "sum"
        )

        true_and_false_positive_TBLRR = (true_positive_TBLRR == 1) & (false_positive_TBLRR == 1)

        true_positive_T = einops.reduce(true_positive_TBLRRC, "T B L R1 R2 C -> T", "sum")
        false_positive_T = einops.reduce(false_positive_TBLRRC, "T B L R1 R2 C -> T", "sum")
        true_negative_T = einops.reduce(true_negative_TBLRRC, "T B L R1 R2 C -> T", "sum")
        false_negative_T = einops.reduce(false_negative_TBLRRC, "T B L R1 R2 C -> T", "sum")
        true_and_false_positive_T = einops.reduce(
            true_and_false_positive_TBLRR, "T B L R1 R2 -> T", "sum"
        )

        batch_size = true_boards_BLRRC.shape[0]
        num_board_states = true_boards_BLRRC.shape[1]
        num_boards = batch_size * num_board_states
        num_squares = int(true_boards_BLRRC.sum().item())

        results[custom_function.__name__]["num_boards"] += num_boards
        results[custom_function.__name__]["num_squares"] += num_squares
        results[custom_function.__name__]["num_true_positive_squares"] += true_positive_T
        results[custom_function.__name__]["num_false_positive_squares"] += false_positive_T
        results[custom_function.__name__]["num_true_negative_squares"] += true_negative_T
        results[custom_function.__name__]["num_false_negative_squares"] += false_negative_T
        results[custom_function.__name__]["num_multiple_classes"] += multiple_classes_T
        results[custom_function.__name__][
            "num_true_and_false_positive_squares"
        ] += true_and_false_positive_T

    return results


def normalize_results(results: dict, n_inputs: int, custom_functions: list[Callable]) -> dict:
    if n_inputs == 0:
        raise ValueError("n_inputs must be greater than 0")
    results["active_per_token"] /= n_inputs
    for custom_function in custom_functions:
        results[custom_function.__name__]["classifiers_per_token"] /= n_inputs
        results[custom_function.__name__]["classified_per_token"] /= n_inputs

    return results


def calculate_F1_scores(
    results: dict, custom_functions: list[Callable], device: torch.device
) -> dict:

    epsilon = 1e-8

    for custom_function in custom_functions:
        num_true_positive_squares_T = results[custom_function.__name__]["num_true_positive_squares"]
        num_false_positive_squares_T = results[custom_function.__name__][
            "num_false_positive_squares"
        ]
        num_positives_T = num_true_positive_squares_T + num_false_positive_squares_T

        false_negatives_T = results[custom_function.__name__]["num_false_negative_squares"]

        precision_T = num_true_positive_squares_T / (num_positives_T + epsilon)
        recall_T = num_true_positive_squares_T / (
            num_true_positive_squares_T + false_negatives_T + epsilon
        )

        # Calculate F1 score
        f1_scores_T = 2 * (precision_T * recall_T) / (precision_T + recall_T + epsilon)
        results[custom_function.__name__]["precision_per_class"] = precision_T
        results[custom_function.__name__]["recall_per_class"] = recall_T
        results[custom_function.__name__]["f1_score_per_class"] = f1_scores_T

        num_true_and_false_positive_squares_T = results[custom_function.__name__][
            "num_true_and_false_positive_squares"
        ]

        # Any square with both true and false positives is a false positive
        adjusted_true_positive_squares_T = (
            num_true_positive_squares_T - num_true_and_false_positive_squares_T
        )
        adjusted_positives_T = adjusted_true_positive_squares_T + num_false_positive_squares_T

        precision_T = adjusted_true_positive_squares_T / (adjusted_positives_T + epsilon)
        recall_T = adjusted_true_positive_squares_T / (
            adjusted_true_positive_squares_T + false_negatives_T + epsilon
        )

        f1_scores_T = 2 * (precision_T * recall_T) / (precision_T + recall_T + epsilon)
        results[custom_function.__name__]["precision_per_square"] = precision_T
        results[custom_function.__name__]["recall_per_square"] = recall_T
        results[custom_function.__name__]["f1_score_per_square"] = f1_scores_T
    return results


def print_out_results(results: dict, custom_functions: list[Callable]):
    print(f"active_per_token", results["active_per_token"])
    print(f"Num alive features: {results['alive_features'].shape[0]}")

    for custom_function in custom_functions:
        print(custom_function.__name__)
        for key in results[custom_function.__name__]:
            print(key, results[custom_function.__name__][key])

        best_idx = results[custom_function.__name__]["f1_score_per_square"].argmax()
        f1 = results[custom_function.__name__]["f1_score_per_square"][best_idx]
        precision = results[custom_function.__name__]["precision_per_square"][best_idx]
        recall = results[custom_function.__name__]["recall_per_square"][best_idx]
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
    data: dict,
    othello: bool = False,
    print_results: bool = False,
    save_results: bool = True,
    precomputed: bool = True,
) -> dict:

    torch.set_grad_enabled(False)
    feature_batch_size = batch_size

    data, ae_bundle, pgn_strings_bL, encoded_inputs_bL = eval_sae.prep_data_ae_buffer_and_model(
        autoencoder_path,
        batch_size,
        data,
        device,
        n_inputs,
        include_buffer=False,
    )

    thresholds_Tf11 = feature_labels["thresholds"].to(device)
    alive_features_f = feature_labels["alive_features"].to(device)
    num_features = len(alive_features_f)
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
        custom_functions, thresholds_Tf11.shape[0], alive_features_f, device
    )

    n_iters = n_inputs // batch_size
    # We round up to ensure we don't ignore the remainder of features
    num_feature_iters = math.ceil(num_features / feature_batch_size)

    for i in tqdm(range(n_iters), desc="Aggregating statistics"):
        start = i * batch_size
        end = (i + 1) * batch_size
        pgn_strings_BL = pgn_strings_bL[start:end]
        encoded_inputs_BL = encoded_inputs_bL[start:end]
        encoded_inputs_BL = torch.tensor(encoded_inputs_BL).to(device)

        batch_data = eval_sae.get_data_batch(
            data,
            pgn_strings_BL,
            start,
            end,
            custom_functions,
            device,
            precomputed=precomputed,
            othello=othello,
        )

        all_activations_fBL, encoded_token_inputs = collect_activations_batch(
            ae_bundle, encoded_inputs_BL, alive_features_f
        )

        if indexing_function is not None:
            all_activations_fBL, batch_data = eval_sae.apply_indexing_function(
                pgn_strings_bL[start:end],
                all_activations_fBL,
                batch_data,
                device,
                indexing_function,
            )

        constructed_boards = initialized_constructed_boards_dict(
            custom_functions, batch_data, thresholds_Tf11, device
        )

        # For thousands of features, this would be many GB of memory. So, we minibatch.
        for feature in range(num_feature_iters):
            f_start = feature * feature_batch_size
            f_end = min((feature + 1) * feature_batch_size, num_features)
            f_batch_size = f_end - f_start

            activations_FBL = all_activations_fBL[f_start:f_end]

            results, additive_boards = aggregate_feature_labels(
                results,
                feature_labels,
                custom_functions,
                activations_FBL,
                thresholds_Tf11[:, f_start:f_end, :, :],
                f_start,
                f_end,
                device,
            )

            for custom_function in constructed_boards:
                constructed_boards[custom_function] += additive_boards[custom_function]
        results = compare_constructed_to_true_boards(
            results, custom_functions, constructed_boards, batch_data, device
        )

    hyperparameters = {"n_inputs": n_inputs}
    results["hyperparameters"] = hyperparameters
    results = normalize_results(results, n_inputs, custom_functions)
    results = calculate_F1_scores(results, custom_functions, device)

    if print_results:
        print_out_results(results, custom_functions)

    if save_results:
        results = to_device(results, "cpu")
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        results = to_device(results, device)
    return results
