from dataclasses import dataclass
import torch

import circuits.chess_utils as chess_utils
import circuits.othello_utils as othello_utils
import circuits.utils as utils

# NOTE: A major baked in assumption is that d_model == 512
# If activation_dim is 512, then the input is the residual stream
# If it's 2048, then the input is the mlp activations


@dataclass
class Config:
    N_GPUS: int = 1  # If you increase this `full_pipeline.py` will use all GPUs
    eval_sae_n_inputs: int = 1000
    batch_size: int = 25  # Reduce this if you run out of memory
    eval_results_n_inputs: int = 1000
    board_reconstruction_n_inputs: int = 1000

    # A feature is a high precision classifier if precision is above the high threshold
    analysis_high_threshold: float = 0.95
    # Low threshold is currently not used
    analysis_low_threshold: float = 0.1
    # A feature must fire on at least this many examples to be considered
    analysis_significance_threshold: int = 10

    # If you have already ran the full pipeline and want to skip steps, you can set these to False
    # If False, they will load the results from the pickle files from the previous run
    run_eval_results: bool = True
    run_eval_sae: bool = True
    run_analysis: bool = True
    run_board_reconstruction: bool = True

    # Saves results.pkl, reconstruction.pkl, and eval_results.pkl
    save_results: bool = True
    # Feature labels use hundreds of MB of disk and can be computed in seconds from results.pkl
    save_feature_labels: bool = False
    # Precompute will create both datasets and save them as pickle files
    # If precompute == False, it creates the dataset on the fly
    # This is far slower when evaluating multiple SAEs, but for an exploratory run it is fine
    precompute: bool = True
    # Analysis on CPU significantly reduces peak memory usage and isn't much slower
    analysis_on_cpu: bool = False
    f1_analysis_thresholds = torch.arange(0.0, 1.1, 0.1)

    # You can speed up runtime by commenting out the functions you don't need
    # All of the functions below will be analyzed in the full pipeline
    othello_functions = [
        othello_utils.games_batch_to_state_stack_mine_yours_BLRRC,
        othello_utils.games_batch_to_state_stack_mine_yours_blank_mask_BLRRC,
        othello_utils.games_batch_to_valid_moves_BLRRC,
        # othello_utils.games_batch_to_state_stack_lines_mine_BLRCC,
        # othello_utils.games_batch_to_state_stack_length_lines_mine_BLRCC,
        # othello_utils.games_batch_to_state_stack_opponent_length_lines_mine_BLRCC,
        # othello_utils.games_batch_to_state_stack_lines_yours_BLRCC,
    ]
    chess_functions = [
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
