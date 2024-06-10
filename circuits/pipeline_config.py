from dataclasses import dataclass
import torch

import circuits.chess_utils as chess_utils
import circuits.othello_utils as othello_utils
import circuits.utils as utils


@dataclass
class Config:
    N_GPUS: int = 1
    eval_sae_n_inputs: int = 1000
    batch_size: int = 25
    eval_results_n_inputs: int = 1000
    board_reconstruction_n_inputs: int = 1000
    analysis_high_threshold: float = 0.95
    analysis_low_threshold: float = 0.1
    analysis_significance_threshold: int = 10
    submodule_type: utils.SubmoduleType = utils.SubmoduleType.resid_post
    run_eval_results: bool = True
    run_eval_sae: bool = True
    run_analysis: bool = True
    run_board_reconstruction: bool = True
    save_results: bool = True
    save_feature_labels: bool = False
    use_separate_test_set: bool = True
    precompute: bool = True
    analysis_on_cpu: bool = False
    f1_analysis_thresholds = torch.arange(0.0, 1.1, 0.1)
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
