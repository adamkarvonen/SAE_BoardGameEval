import pandas as pd

import circuits.full_pipeline as full_pipeline
import circuits.pipeline_config as pipeline_config

TEST_INPUTS = 25
BATCH_SIZE = 5
TOLERANCE = 1e-6
PRECOMPUTE = False


def test_full_chess_pipeline():

    test_config = pipeline_config.Config(
        eval_sae_n_inputs=TEST_INPUTS,
        batch_size=BATCH_SIZE,
        eval_results_n_inputs=TEST_INPUTS,
        board_reconstruction_n_inputs=TEST_INPUTS,
        analysis_on_cpu=True,
        precompute=PRECOMPUTE,
    )
    autoencoder_group_paths = ["autoencoders/testing_chess"]
    csv_output_path = f"{autoencoder_group_paths[0]}/results.csv"
    final_output_path = f"{autoencoder_group_paths[0]}/f1_results.csv"

    full_pipeline.analyze_sae_groups(autoencoder_group_paths, csv_output_path, test_config)

    df = pd.read_csv(final_output_path)

    actual_result = df[
        "board_to_piece_masked_blank_and_initial_state_best_f1_score_per_class"
    ].max()

    expected_result = 0.7595600485801697

    assert actual_result - expected_result < TOLERANCE


def test_full_othello_pipeline():

    test_config = pipeline_config.Config(
        eval_sae_n_inputs=TEST_INPUTS,
        batch_size=BATCH_SIZE,
        eval_results_n_inputs=TEST_INPUTS,
        board_reconstruction_n_inputs=TEST_INPUTS,
        analysis_on_cpu=True,
        precompute=PRECOMPUTE,
    )
    autoencoder_group_paths = ["autoencoders/testing_othello"]
    csv_output_path = f"{autoencoder_group_paths[0]}/results.csv"
    final_output_path = f"{autoencoder_group_paths[0]}/f1_results.csv"

    full_pipeline.analyze_sae_groups(autoencoder_group_paths, csv_output_path, test_config)

    df = pd.read_csv(final_output_path)

    actual_result = df[
        "games_batch_to_state_stack_mine_yours_blank_mask_BLRRC_best_f1_score_per_class"
    ].max()

    expected_result = 0.928957581520081

    assert actual_result - expected_result < TOLERANCE
