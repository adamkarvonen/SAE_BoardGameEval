import pickle
import torch
from typing import Callable
import einops
import chess
import os

import circuits.chess_utils as chess_utils
from circuits.utils import to_cpu
from circuits.eval_sae_as_classifier import normalize_tracker


def get_all_file_names(folder_name: str) -> list[str]:
    """Get all file names with the .pkl extension in the given folder."""
    file_names = []
    for file_name in os.listdir(folder_name):
        if file_name.endswith(".pkl"):
            file_names.append(file_name)
    return file_names


def get_above_below_counts(
    tracker_TF: torch.Tensor,
    counts_TF: torch.Tensor,
    low_threshold: float,
    high_threshold: float,
    significance_threshold: int = 10,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Must be a 2D tensor matching shape annotation."""

    # Find all elements that were active more than x% of the time (high_threshold)
    above_freq_TF_mask = tracker_TF >= high_threshold

    # For the counts tensor, zero out all elements that were not active enough
    above_counts_TF = counts_TF * above_freq_TF_mask

    # Find all features that were active more than significance_threshold times
    above_counts_TF_mask = above_counts_TF >= significance_threshold

    # Zero out all elements that were not active enough
    above_counts_TF = above_counts_TF * above_counts_TF_mask

    # Count the number of elements that were active more than high_threshold % and significance_threshold times
    above_counts_T = above_counts_TF_mask.sum(dim=(1))

    # All nonzero elements are set to 1
    above_counts_TF = (above_counts_TF != 0).int()

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

    return above_counts_T, above_counts_TF


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


def analyze_board_tracker(
    results: dict,
    function: str,
    key: str,
    device: torch.device,
    high_threshold: float,
    significance_threshold: int,
    mine_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the board tracker for analysis."""
    normalized_key = key + "_normalized"

    num_thresholds = results[function][normalized_key].shape[0]

    piece_state_on_normalized = results[function][normalized_key].clone().view(num_thresholds, -1)
    piece_state_on = results[function][key].clone()
    original_shape = piece_state_on.shape

    piece_state_on = mask_initial_board_state(piece_state_on, device, mine_state)

    # Optionally, we also mask off the blank class
    piece_state_on[:, :, :, :, 6] = 0

    # Flatten the tensor to a 2D shape for compatibility with get_above_below_counts()
    piece_state_on = piece_state_on.view(num_thresholds, -1)

    above_counts_T, above_counts_TF = get_above_below_counts(
        piece_state_on_normalized,
        piece_state_on,
        0.00,
        high_threshold,
        significance_threshold=significance_threshold,
    )

    best_idx = torch.argmax(above_counts_T)

    above_counts_TFRRC = above_counts_TF.view(original_shape)

    best_counts_FRRC = above_counts_TFRRC[best_idx, ...]

    summary_board_RR = einops.reduce(best_counts_FRRC, "F R1 R2 C -> R1 R2", "sum").to(torch.int)

    class_dict_C = einops.reduce(best_counts_FRRC, "F R1 R2 C -> C", "sum").to(torch.int)

    return above_counts_T, summary_board_RR, class_dict_C


if __name__ == "__main__":
    folder_name = "layer5_large_sweep_results2/"
    folder_name = "layer0_results/"
    file_names = get_all_file_names(folder_name)
    device = torch.device("cpu")

    high_threshold = 0.95
    significance_threshold = 10

    for file_name in file_names:
        print()
        print(file_name)
        with open(folder_name + file_name, "rb") as file:
            results = pickle.load(file)
            results = to_cpu(results)

        results = normalize_tracker(
            results,
            "on",
            [chess_utils.board_to_pin_state, chess_utils.board_to_piece_state],
            device,
        )

        results = normalize_tracker(
            results,
            "off",
            [chess_utils.board_to_pin_state, chess_utils.board_to_piece_state],
            device,
        )

        above_counts_T, above_counts_TF = get_above_below_counts(
            results["board_to_pin_state"]["on_normalized"].squeeze().clone(),
            results["board_to_pin_state"]["on"].squeeze().clone(),
            0.00,
            high_threshold,
            significance_threshold=significance_threshold,
        )

        piece_state_above_counts_T, summary_board, class_dict = analyze_board_tracker(
            results,
            "board_to_piece_state",
            "on",
            device,
            high_threshold,
            significance_threshold,
        )

        print("Number of alive features:")
        print(above_counts_TF.shape[1])
        print("Pin state:")
        print(above_counts_T)
        print()
        print("Piece state:")
        print(piece_state_above_counts_T)
        print(summary_board)
        print(class_dict)

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
