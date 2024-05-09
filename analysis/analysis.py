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
        if file_name.endswith(".pkl"):
            file_names.append(file_name)
    return file_names


def get_above_below_counts(
    on_tracker_TF: torch.Tensor,
    on_counts_TF: torch.Tensor,
    off_tracker_TF: torch.Tensor,
    off_counts_TF: torch.Tensor,
    low_threshold: float,
    high_threshold: float,
    significance_threshold: int = 10,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Must be a 2D tensor matching shape annotation."""

    # Find all elements that were active more than x% of the time (high_threshold)
    above_freq_TF_mask = on_tracker_TF >= high_threshold

    # Find all elements that were active less than x% of the time (low_threshold)
    below_freq_TF_mask = off_tracker_TF <= low_threshold

    # For the counts tensor, zero out all elements that were not active enough
    above_counts_TF = on_counts_TF * above_freq_TF_mask

    # Find all features that were active more than significance_threshold times
    # This isn't required for the off_counts_TF tensor
    above_counts_TF_mask = above_counts_TF >= significance_threshold

    # Zero out all elements that were not active enough. Now we have elements that
    # were active more than high_threshold % and significance_threshold times (high precision)
    # But, this doesn't say anything about recall
    above_counts_TF = above_counts_TF * above_counts_TF_mask

    # Zero out all elements that were active when the feature was off. Now, this is
    # a classifier that is high precision and high recall
    classifier_TF = above_counts_TF * below_freq_TF_mask

    # All nonzero elements are set to 1
    above_counts_TF = (above_counts_TF != 0).int()
    classifier_TF = (classifier_TF != 0).int()

    # Count the number of elements that were active more than high_threshold % and significance_threshold times
    above_counts_T = above_counts_TF.sum(dim=(1))
    classifier_counts_T = classifier_TF.sum(dim=(1))

    if verbose:
        print(
            f"\nThis is the number of elements that were active more than {high_threshold} and {significance_threshold} times."
        )
        print(
            f"Note that this shape is num_thresholds, and every element corresponds to a threshold."
        )
        print(above_counts_T)

        above_T = above_freq_TF_mask.sum(dim=(1))

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

    return above_counts_T, above_counts_TF, classifier_counts_T, classifier_TF


def transform_board_from_piece_color_to_piece(board: torch.Tensor) -> torch.Tensor:
    new_board = torch.zeros(board.shape[:-1] + (7,), dtype=board.dtype, device=board.device)

    for i in range(7):
        if i == 6:
            new_board[..., i] = board[..., 6]
        else:
            new_board[..., i] = board[..., i] + board[..., 12 - i]
    return new_board


def mask_initial_board_state(
    on_tracker: torch.Tensor, device: torch.device, mine_state: bool = False
) -> torch.Tensor:
    initial_board = chess.Board()
    initial_state = chess_utils.board_to_piece_state(initial_board)
    initial_state = initial_state.view(1, 1, 8, 8)
    initial_one_hot = chess_utils.state_stack_to_one_hot(
        chess_utils.piece_config, device, initial_state
    ).squeeze()

    if mine_state:
        initial_one_hot = transform_board_from_piece_color_to_piece(initial_one_hot)

    mask = initial_one_hot == 1
    on_tracker[:, :, mask] = 0

    return on_tracker


def get_summary_board(
    above_counts_T: torch.Tensor, above_counts_TF: torch.Tensor, original_shape: tuple[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    best_idx = torch.argmax(above_counts_T)

    above_counts_TFRRC = above_counts_TF.view(original_shape)

    best_counts_FRRC = above_counts_TFRRC[best_idx, ...]

    summary_board_RR = einops.reduce(best_counts_FRRC, "F R1 R2 C -> R1 R2", "sum").to(torch.int)

    class_dict_C = einops.reduce(best_counts_FRRC, "F R1 R2 C -> C", "sum").to(torch.int)

    return summary_board_RR, class_dict_C


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the board tracker for analysis."""

    othello = False

    if function in othello_utils.othello_functions:
        othello = True

    normalized_on_key = on_key + "_normalized"
    normalized_off_key = off_key + "_normalized"

    num_thresholds = results[function][normalized_on_key].shape[0]

    piece_state_on_normalized = (
        results[function][normalized_on_key].clone().view(num_thresholds, -1)
    )
    piece_state_off_normalized = (
        results[function][normalized_off_key].clone().view(num_thresholds, -1)
    )
    piece_state_on = results[function][on_key].clone()
    piece_state_off = results[function][off_key].clone()
    original_shape = piece_state_on.shape

    if not othello:
        piece_state_on = mask_initial_board_state(piece_state_on, device, mine_state)
        # Optionally, we also mask off the blank class
        piece_state_on[:, :, :, :, 6] = 0
    else:
        # Optionally, we also mask off the blank class
        piece_state_on[:, :, :, :, 1] = 0

    # Flatten the tensor to a 2D shape for compatibility with get_above_below_counts()
    piece_state_on = piece_state_on.view(num_thresholds, -1)
    piece_state_off = piece_state_off.view(num_thresholds, -1)

    above_counts_T, above_counts_TF, classifier_counts_T, classifier_TF = get_above_below_counts(
        piece_state_on_normalized,
        piece_state_on,
        piece_state_off_normalized,
        piece_state_off,
        low_threshold,
        high_threshold,
        significance_threshold=significance_threshold,
    )

    summary_board_RR, class_dict_C = get_summary_board(
        above_counts_T, above_counts_TF, original_shape
    )

    classifier_summary_board_RR, classifier_class_dict_C = get_summary_board(
        classifier_counts_T, classifier_TF, original_shape
    )

    return (
        above_counts_T,
        summary_board_RR,
        class_dict_C,
        classifier_counts_T,
        classifier_summary_board_RR,
        classifier_class_dict_C,
    )


if __name__ == "__main__":
    folder_name = "layer5_large_sweep_results2/"
    # folder_name = "layer0_results/"
    # folder_name = "layer5_indexing_results/"
    folder_name = "layer5_large_sweep_indexing_results/"
    # folder_name = "group1_results/"
    # folder_name = "before_after_compare/"
    # folder_name = "othello_results/"
    # folder_name = "othello_layer5_even_index/"
    # folder_name = "othello_mine_yours_results/"
    # folder_name = "othello_layer0_results/"
    file_names = get_all_file_names(folder_name)
    device = torch.device("cpu")

    high_threshold = 0.98
    low_threshold = 0.1
    significance_threshold = 20

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

        for custom_function in custom_functions:
            func_name = custom_function.__name__
            config = chess_utils.config_lookup[func_name]
            if config.num_rows == 8:
                (
                    piece_state_above_counts_T,
                    summary_board,
                    class_dict,
                    classifier_counts_T,
                    classifier_summary_board,
                    classifier_class_dict,
                ) = analyze_board_tracker(
                    results,
                    func_name,
                    "on",
                    "off",
                    device,
                    high_threshold,
                    low_threshold,
                    significance_threshold,
                )

                print("Piece state (high precision):")
                print(piece_state_above_counts_T)
                print(summary_board)
                print(class_dict)
                print()
                print(f"{func_name} (high precision and recall):")
                print(classifier_counts_T)
                print(classifier_summary_board)
                print(classifier_class_dict)
                print()
            else:
                above_counts_T, above_counts_TF, classifier_counts_T, classifier_counts_TF = (
                    get_above_below_counts(
                        results[func_name]["on_normalized"].squeeze().clone(),
                        results[func_name]["on"].squeeze().clone(),
                        results[func_name]["off_normalized"].squeeze().clone(),
                        results[func_name]["off"].squeeze().clone(),
                        low_threshold,
                        high_threshold,
                        significance_threshold=significance_threshold,
                    )
                )

                print(f"{func_name} (high precision):")
                print(above_counts_T)
                print()
                print(f"{func_name} (high precision and recall):")
                print(classifier_counts_T)
                print()

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
