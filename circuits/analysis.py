import pickle
import torch
from typing import Callable
import einops
import chess
import os

import circuits.chess_utils as chess_utils
import circuits.othello_utils as othello_utils
from circuits.utils import to_device
from circuits.eval_sae_as_classifier import normalize_tracker


def get_all_file_names(folder_name: str) -> list[str]:
    """Get all file names with the .pkl extension in the given folder."""
    file_names = []
    for file_name in os.listdir(folder_name):
        if file_name.endswith(".pkl") and "feature_labels" not in file_name:
            file_names.append(file_name)
    return file_names


def get_above_below_counts(
    on_tracker_TFRRC: torch.Tensor,
    on_counts_TFRRC: torch.Tensor,
    off_tracker_TFRRC: torch.Tensor,
    off_counts_TFRRC: torch.Tensor,
    low_threshold: float,
    high_threshold: float,
    significance_threshold: int = 10,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """on_tracker_TFRRC: Every element is this: What percentage of the time was this piece on this square
    when the feature was active above the threshold? If the feature was active 100% of the time, this is 1.
    If the feature is a high precision classifier, this is 1.
    If it is high recall, the corresponding off_tracker_TFRRC value is 0."""

    # Find all elements that were active more than x% of the time (high_threshold)
    above_freq_TFRRC_mask = on_tracker_TFRRC >= high_threshold

    # Find all elements that were active less than x% of the time (low_threshold)
    below_freq_TFRRC_mask = off_tracker_TFRRC <= low_threshold

    # For the counts tensor, zero out all elements that were not active enough
    above_counts_TFRRC = on_counts_TFRRC * above_freq_TFRRC_mask

    # Find all features that were active more than significance_threshold times
    # This isn't required for the off_counts_TF tensor
    above_counts_TFRRC_mask = above_counts_TFRRC >= significance_threshold

    # Zero out all elements that were not active enough. Now we have elements that
    # were active more than high_threshold % and significance_threshold times (high precision)
    # But, this doesn't say anything about recall
    above_counts_TFRRC = above_counts_TFRRC * above_counts_TFRRC_mask

    # Zero out all elements that were active when the feature was off. Now, this is
    # a classifier that is high precision and high recall
    classifier_TFRRC = above_counts_TFRRC * below_freq_TFRRC_mask

    # All nonzero elements are set to 1
    above_counts_TFRRC_binary = (above_counts_TFRRC != 0).int()
    classifier_TFRRC_binary = (classifier_TFRRC != 0).int()

    # Count the number of elements that were active more than high_threshold % and significance_threshold times
    above_counts_T = einops.reduce(above_counts_TFRRC_binary, "T F R1 R2 C -> T", "sum")
    classifier_counts_T = einops.reduce(classifier_TFRRC_binary, "T F R1 R2 C -> T", "sum")

    if verbose:
        print(
            f"\nThis is the number of elements that were active more than {high_threshold} and {significance_threshold} times."
        )
        print(
            f"Note that this shape is num_thresholds, and every element corresponds to a threshold."
        )
        print(above_counts_T)

        above_T = einops.reduce(above_freq_TFRRC_mask, "T F R1 R2 C -> T", "sum")

        print(
            f"\nThis is the number of elements that were active more than {high_threshold} percent."
        )
        print(above_T)

    # Count the number of elements that were active less than low_threshold %
    # below_T = below_freq_TF_mask.sum(dim=(1))
    # # Count the number of elements that were active more than high_threshold %
    # above_T = above_freq_TF_mask.sum(dim=(1))

    # values_above_threshold = [tracker_TF[i, above_freq_TF_mask[i]] for i in range(tracker_TF.size(0))]
    # counts_above_threshold = [counts_TF[i, above_freq_TF_mask[i]] for i in range(tracker_TF.size(0))]

    # for i, values in enumerate(values_above_threshold):
    #     print(f"Row {i} values above {high_threshold}: {values.tolist()}")

    # for i, counts in enumerate(counts_above_threshold):
    #     print(f"Row {i} counts above {high_threshold}: {counts.tolist()}")

    return (
        above_counts_T,
        above_counts_TFRRC_binary,
        above_counts_TFRRC,
        classifier_counts_T,
        classifier_TFRRC_binary,
        classifier_TFRRC,
    )


def transform_board_from_piece_color_to_piece(board: torch.Tensor) -> torch.Tensor:
    new_board = torch.zeros(board.shape[:-1] + (7,), dtype=board.dtype, device=board.device)

    for i in range(7):
        if i == 6:
            new_board[..., i] = board[..., 6]
        else:
            new_board[..., i] = board[..., i] + board[..., 12 - i]
    return new_board


def mask_initial_board_state(
    on_tracker_TFRRC: torch.Tensor, device: torch.device, mine_state: bool = False
) -> torch.Tensor:
    initial_board = chess.Board()
    initial_state_RRC = chess_utils.board_to_piece_state(initial_board)
    initial_state_11RRC = einops.rearrange(initial_state_RRC, "R1 R2 -> 1 1 R1 R2")
    initial_one_hot_11RRC = chess_utils.state_stack_to_one_hot(
        chess_utils.piece_config, device, initial_state_11RRC
    ).squeeze()

    if mine_state:
        initial_one_hot_11RRC = transform_board_from_piece_color_to_piece(initial_one_hot_11RRC)

    mask_11RRC = initial_one_hot_11RRC == 1
    on_tracker_TFRRC[:, :, mask_11RRC] = 0

    return on_tracker_TFRRC


def get_summary_board(
    above_counts_T: torch.Tensor, above_counts_TFRRC: torch.Tensor, original_shape: tuple[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    best_idx = torch.argmax(above_counts_T)

    above_counts_TFRRC = above_counts_TFRRC

    best_counts_FRRC = above_counts_TFRRC[best_idx, ...]

    summary_board_RR = einops.reduce(best_counts_FRRC, "F R1 R2 C -> R1 R2", "sum").to(torch.int)

    class_dict_C = einops.reduce(best_counts_FRRC, "F R1 R2 C -> C", "sum").to(torch.int)

    coverage_RRC = einops.reduce(best_counts_FRRC, "F R1 R2 C -> R1 R2 C", "sum").to(torch.int)

    coverage_RRC = coverage_RRC > 0

    coverage_RR = einops.reduce(coverage_RRC, "R1 R2 C -> R1 R2", "sum").to(torch.int)

    coverage = int(coverage_RR.sum().item())

    return summary_board_RR, class_dict_C, coverage_RR, coverage


def get_timestep_counts(
    above_counts_TFRRC: torch.Tensor,
    piece_state_on_TFRRC: torch.Tensor,
    piece_state_off_counting_TFRRC: torch.Tensor,
):
    """Get the counts for the total number of times each piece type was present on a square over all time steps.
    Get the counts for the total number of times a feature classifier predicted a piece type was present on a square
    over all time steps. If the SAE was perfect, these two counts should be equal."""
    total_counts_TFRRC = piece_state_on_TFRRC + piece_state_off_counting_TFRRC

    class_counts_C = einops.reduce(total_counts_TFRRC[0][0], "R1 R2 C -> C", "sum")
    class_counts_C2 = einops.reduce(total_counts_TFRRC[1][2], "R1 R2 C -> C", "sum")

    assert torch.equal(total_counts_TFRRC[0][0], total_counts_TFRRC[1][2])
    assert torch.equal(class_counts_C, class_counts_C2)

    # print(f"Class counts: {class_counts_C}")

    # above_counts_FRRC = above_counts_TFRRC[best_idx, ...]
    actual_class_counts_TC = einops.reduce(above_counts_TFRRC, "T F R1 R2 C -> T C", "sum")
    actual_class_counts_T = einops.reduce(actual_class_counts_TC, "T C -> T", "sum")

    best_idx = torch.argmax(actual_class_counts_T)
    actual_class_counts_C = actual_class_counts_TC[best_idx, ...]

    # print(f"Actual class counts: {actual_class_counts_C}")

    class_counts_C[class_counts_C == 0] += 1  # Avoid division by zero
    coverage2_C = actual_class_counts_C / class_counts_C
    coverage2 = coverage2_C.sum().item() / coverage2_C.size(0)

    print(f"Percent coverage over time dimension: {coverage2}")


def analyze_board_tracker(
    results: dict,
    function: str,
    on_key: str,
    off_key: str,
    device: torch.device,
    high_threshold: float,
    low_threshold: float,
    significance_threshold: int,
    mine_state: bool = False,
) -> torch.Tensor:

    othello = False

    if function in othello_utils.othello_functions:
        othello = True

    normalized_on_key = on_key + "_normalized"
    normalized_off_key = off_key + "_normalized"

    num_thresholds = results[function][normalized_on_key].shape[0]

    piece_state_on_normalized = results[function][normalized_on_key].clone()
    piece_state_off_normalized = results[function][normalized_off_key].clone()
    piece_state_on_TFRRC = results[function][on_key].clone()
    piece_state_off_TFRRC = results[function][off_key].clone()
    original_shape = piece_state_on_TFRRC.shape

    piece_state_off_counting_TFRRC = piece_state_off_TFRRC.clone()

    if not othello:
        piece_state_on_TFRRC = mask_initial_board_state(piece_state_on_TFRRC, device, mine_state)
        piece_state_off_counting_TFRRC = mask_initial_board_state(
            piece_state_off_counting_TFRRC, device, mine_state
        )
        # Optionally, we also mask off the blank class
        piece_state_on_TFRRC[:, :, :, :, 6] = 0
        piece_state_off_counting_TFRRC[:, :, :, :, 6] = 0
    else:
        # Optionally, we also mask off the blank class
        piece_state_on_TFRRC[:, :, :, :, 1] = 0
        piece_state_off_counting_TFRRC[:, :, :, :, 1] = 0

    (
        above_counts_T,
        above_counts_binary_TFRRC,
        above_counts_TFRRC,
        classifier_counts_T,
        classifier_counts_binary_TFRRC,
        classifier_counts_TFRRC,
    ) = get_above_below_counts(
        piece_state_on_normalized,
        piece_state_on_TFRRC,
        piece_state_off_normalized,
        piece_state_off_TFRRC,
        low_threshold,
        high_threshold,
        significance_threshold=significance_threshold,
    )

    print("\nTime coverage for high precision:")
    get_timestep_counts(above_counts_TFRRC, piece_state_on_TFRRC, piece_state_off_counting_TFRRC)
    print("\nTime coverage for high precision and recall:")
    get_timestep_counts(
        classifier_counts_TFRRC,
        piece_state_on_TFRRC,
        piece_state_off_counting_TFRRC,
    )

    summary_board_RR, class_dict_C, coverage_RR, coverage = get_summary_board(
        above_counts_T, above_counts_binary_TFRRC, original_shape
    )

    (
        classifier_summary_board_RR,
        classifier_class_dict_C,
        classifier_coverage_RR,
        classifier_coverage,
    ) = get_summary_board(classifier_counts_T, classifier_counts_binary_TFRRC, original_shape)

    # -1 because we mask off blank
    max_possible_coverage = (
        summary_board_RR.shape[0] * summary_board_RR.shape[1] * (class_dict_C.shape[0] - 1)
    )

    print(
        f"{func_name} (high precision) coverage {coverage} out of {max_possible_coverage} max possible:"
    )
    print(above_counts_T)
    print(summary_board_RR)
    print(class_dict_C)
    print(coverage_RR)
    print()
    print(
        f"{func_name} (high precision and recall) coverage {classifier_coverage} out of {max_possible_coverage} max possible::"
    )
    print(classifier_counts_T)
    print(classifier_summary_board_RR)
    print(classifier_class_dict_C)
    print(classifier_coverage_RR)
    print()

    return above_counts_binary_TFRRC


if __name__ == "__main__":
    folder_name = "layer5_large_sweep_results2/"
    folder_name = "layer0_results/"
    folder_name = "layer5_indexing_results/"
    # folder_name = "layer5_large_sweep_indexing_results/"
    # folder_name = "group1_results/"
    # folder_name = "before_after_compare/"
    # folder_name = "othello_results/"
    # folder_name = "othello_layer5_even_index/"
    # folder_name = "othello_mine_yours_results/"
    # folder_name = "othello_layer0_results/"
    # folder_name = "othello_even_no_last_move_results/"
    # folder_name = "othello_layer0_no_last_move/"
    # folder_name = "autoencoders_othello_layer5_ef4__indexing_None_results/"
    # folder_name = "autoencoders_othello_layer0__indexing_None_results/"
    # folder_name = "autoencoders_chess_layer_0_subset__indexing_find_dots_indices_results/"
    file_names = get_all_file_names(folder_name)
    device = torch.device("cpu")

    high_threshold = 0.95
    low_threshold = 0.1
    significance_threshold = 10

    for file_name in file_names:
        print()
        print(file_name)
        with open(folder_name + file_name, "rb") as file:
            results = pickle.load(file)
            results = to_device(results, device)

        custom_functions = []

        for key in results:
            if key in chess_utils.config_lookup:
                custom_functions.append(chess_utils.config_lookup[key].custom_board_state_function)

        results = normalize_tracker(
            results,
            "on",
            custom_functions,
            device,
        )

        results = normalize_tracker(
            results,
            "off",
            custom_functions,
            device,
        )

        print("Number of alive features:")
        print(results["on_count"].shape[1])

        print("Number of inputs:")
        print(results["hyperparameters"]["n_inputs"])

        thresholds_TF11 = results["hyperparameters"]["thresholds"]
        feature_labels = {
            "thresholds": thresholds_TF11,
            "alive_features": results["alive_features"],
            "indexing_function": results["hyperparameters"]["indexing_function"],
        }

        for custom_function in custom_functions:
            func_name = custom_function.__name__
            config = chess_utils.config_lookup[func_name]
            if config.num_rows == 8:
                above_counts_binary_TFRRC = analyze_board_tracker(
                    results,
                    func_name,
                    "on",
                    "off",
                    device,
                    high_threshold,
                    low_threshold,
                    significance_threshold,
                )
                feature_labels[func_name] = above_counts_binary_TFRRC

            else:
                (
                    above_counts_T,
                    above_counts_binary_TFRRC,
                    above_counts_TFRRC,
                    classifier_counts_T,
                    classifier_counts_binary_TFRRC,
                    classifier_counts_TFRRC,
                ) = get_above_below_counts(
                    results[func_name]["on_normalized"].clone(),
                    results[func_name]["on"].clone(),
                    results[func_name]["off_normalized"].clone(),
                    results[func_name]["off"].clone(),
                    low_threshold,
                    high_threshold,
                    significance_threshold=significance_threshold,
                )

                print(f"{func_name} (high precision):")
                print(above_counts_T)
                print()
                print(f"{func_name} (high precision and recall):")
                print(classifier_counts_T)
                print()

        feature_labels_name = file_name.split(".")[0] + "_feature_labels.pkl"
        with open(folder_name + feature_labels_name, "wb") as write_file:
            pickle.dump(feature_labels, write_file)

        # results["board_to_piece_state"]["on_piece"] = transform_board_from_piece_color_to_piece(
        #     results["board_to_piece_state"]["on"]
        # )
        # results["on_piece_count"] = results["on_count"]
        # results = normalize_tracker(results, "on_piece", [chess_utils.board_to_piece_state], device)

        # mine_state_above_counts_T, mine_summary_board, mine_class_dict = analyze_board_tracker(
        #     results,
        #     "board_to_piece_state",
        #     "on_piece",
        #     device,
        #     high_threshold,
        #     significance_threshold,
        #     mine_state=True,
        # )

        # print()
        # print("Piece state (mine):")
        # print(mine_state_above_counts_T)
        # print(mine_summary_board)
        # print(mine_class_dict)
