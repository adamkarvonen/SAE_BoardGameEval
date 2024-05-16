import torch

import circuits.eval_sae_as_classifier as eval_sae_as_classifier
import circuits.chess_utils as chess_utils
import circuits.analysis as analysis
import chess

DEVICE = torch.device("cuda")


def test_initialize_results_dict():
    custom_functions = [chess_utils.board_to_piece_state, chess_utils.board_to_pin_state]
    num_thresholds = 2
    alive_features_F = torch.arange(4).to(DEVICE)

    results = eval_sae_as_classifier.initialize_results_dict(
        custom_functions, num_thresholds, alive_features_F, DEVICE
    )

    assert (
        len(results) == len(custom_functions) + 3
    )  # 3 for on_count, off_count, and alive_features


def test_aggregate_batch_statistics():
    custom_functions = [chess_utils.board_to_piece_state, chess_utils.board_to_pin_state]
    num_features = 4
    alive_features_F = torch.arange(num_features).to(DEVICE)
    thresholds_T111 = (
        torch.arange(0.0, 1.0, 0.5).view(-1, 1, 1, 1).to(DEVICE)
    )  # Reshape for broadcasting

    results = eval_sae_as_classifier.initialize_results_dict(
        custom_functions, len(thresholds_T111), alive_features_F, DEVICE
    )

    batch_size = 3

    inputs_BL = [";1.e4 e5 2.d3 f6 3.Nd2 Bb4 4.a3"] * batch_size
    # there is a pin after Bb4 (5 characters total)

    game_len = len(inputs_BL[0])

    data = {}
    batch_data = eval_sae_as_classifier.get_data_batch(
        data, inputs_BL, 0, batch_size, custom_functions, DEVICE, precomputed=False
    )

    all_activations_FBL = torch.zeros(num_features, batch_size, game_len).to(DEVICE)

    dim_0_on = 5
    all_activations_FBL[0, :, (game_len - dim_0_on) : game_len] = (
        0.6  # Feature 0 activates on pins and final state
    )
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

    dim_0_on_count = dim_0_on * batch_size  # 5.0 == num_chars
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
    board.push_san("e4")
    first_move_state = config.custom_board_state_function(board)
    first_move_state = first_move_state.view(1, 1, config.num_rows, config.num_cols)
    first_move_one_hot = chess_utils.state_stack_to_one_hot(
        config, DEVICE, first_move_state
    ).squeeze()

    assert torch.equal(
        results["board_to_piece_state"]["on"][0, 1, :, :, :], first_move_one_hot * dim_1_on_count
    )

    results = eval_sae_as_classifier.update_all_tracker(
        results, custom_functions, batch_data, DEVICE
    )
    results = analysis.add_off_tracker(results, custom_functions, DEVICE)

    results = eval_sae_as_classifier.normalize_tracker(results, "on", custom_functions, DEVICE)
    results = eval_sae_as_classifier.normalize_tracker(results, "off", custom_functions, DEVICE)

    assert torch.equal(
        results["board_to_piece_state"]["on_normalized"][0, 1, :, :, :], first_move_one_hot
    )

    assert torch.equal(
        results["board_to_pin_state"]["on_normalized"][0, -4, :, :, :].squeeze(),
        torch.tensor(1.0).to(DEVICE),
    )
    assert torch.equal(
        results["board_to_pin_state"]["off_normalized"][0, -4, :, :, :].squeeze(),
        torch.tensor(0.0).to(DEVICE),
    )
    assert results["board_to_pin_state"]["off_normalized"][0, 2, :, :, :].squeeze().item() < 0.5
