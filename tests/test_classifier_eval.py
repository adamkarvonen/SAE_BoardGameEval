import torch

import circuits.eval_sae_as_classifier as eval_sae_as_classifier
import circuits.chess_utils as chess_utils
import chess

DEVICE = torch.device("cuda")


def test_initialize_results_dict():
    custom_functions = [chess_utils.board_to_piece_state, chess_utils.board_to_pin_state]
    num_thresholds = 2
    num_features = 4

    results = eval_sae_as_classifier.initialize_results_dict(
        custom_functions, num_thresholds, num_features, DEVICE
    )

    assert len(results) == len(custom_functions) + 2  # 2 for on_count and off_count


def test_aggregate_batch_statistics():
    custom_functions = [chess_utils.board_to_piece_state, chess_utils.board_to_pin_state]
    # custom_functions = [chess_utils.board_to_pin_state]
    custom_functions = [chess_utils.board_to_piece_state]
    num_features = 4
    thresholds_T111 = (
        torch.arange(0.0, 1.0, 0.5).view(-1, 1, 1, 1).to(DEVICE)
    )  # Reshape for broadcasting

    results = eval_sae_as_classifier.initialize_results_dict(
        custom_functions, len(thresholds_T111), num_features, DEVICE
    )

    batch_size = 3

    inputs_BL = [
        ";1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 d6 8.c3 O-O"
    ] * batch_size

    game_len = len(inputs_BL[0])
    data = {}
    data["board_to_pin_state"] = torch.zeros(batch_size, game_len, 1, 1, 2).to(DEVICE)
    data["board_to_pin_state"][:, 0:5, :, :, 1] = 1  # All games have a pin before the first move

    batch_data = eval_sae_as_classifier.get_data_batch(
        data, inputs_BL, 0, batch_size - 1, custom_functions, DEVICE
    )

    all_activations_FBL = torch.zeros(num_features, batch_size, game_len).to(DEVICE)
    all_activations_FBL[0, :, 0:5] = 0.6  # Feature 0 activates on pins
    all_activations_FBL[1, :, 5:8] = 0.6  # Feature 1 activates on the first move

    start = 0
    f_batch_size = num_features - start
    results = eval_sae_as_classifier.aggregate_batch_statistics(
        results,
        custom_functions,
        all_activations_FBL,
        thresholds_T111,
        batch_data,
        start,
        num_features,
        f_batch_size,
        DEVICE,
    )

    dim_0_on_count = 5.0 * batch_size  # 5.0 == num_chars
    dim_1_on_count = 3.0 * batch_size
    expected_on_count = torch.tensor([dim_0_on_count, dim_1_on_count, 0.0, 0.0]).to(DEVICE)

    assert torch.equal(results["on_count"][0, :], expected_on_count)
    assert torch.equal(results["on_count"][1, :], expected_on_count)

    total_chars = game_len * batch_size
    expected_off_count = torch.tensor([total_chars] * 4).to(DEVICE) - expected_on_count

    assert torch.equal(results["off_count"][0, :], expected_off_count)
    assert torch.equal(results["off_count"][1, :], expected_off_count)

    config = chess_utils.piece_config
    board = chess.Board()
    initial_state = config.custom_board_state_function(board)
    initial_state = initial_state.view(1, 1, config.num_rows, config.num_cols)
    initial_one_hot = chess_utils.state_stack_to_one_hot(config, DEVICE, initial_state).squeeze()

    board.push_san("e4")
    first_move_state = config.custom_board_state_function(board)
    first_move_state = first_move_state.view(1, 1, config.num_rows, config.num_cols)
    first_move_one_hot = chess_utils.state_stack_to_one_hot(
        config, DEVICE, first_move_state
    ).squeeze()

    results = eval_sae_as_classifier.normalize_tracker(results, "on", custom_functions, DEVICE)
    results = eval_sae_as_classifier.normalize_tracker(results, "off", custom_functions, DEVICE)

    assert torch.equal(
        results["board_to_piece_state"]["on_normalized"][0, 0, :, :, :], initial_one_hot
    )
    assert torch.equal(
        results["board_to_piece_state"]["on_normalized"][0, 1, :, :, :], first_move_one_hot
    )
