import pandas as pd
import torch

import circuits.full_pipeline as full_pipeline
import circuits.pipeline_config as pipeline_config
import circuits.eval_sae_as_classifier as eval_sae_as_classifier
import circuits.chess_utils as chess_utils

TEST_INPUTS = 25
BATCH_SIZE = 5
TOLERANCE = 1e-6
PRECOMPUTE = False
DEVICE = "cuda"

TEST_CONFIG = pipeline_config.Config(
    eval_sae_n_inputs=TEST_INPUTS,
    batch_size=BATCH_SIZE,
    eval_results_n_inputs=TEST_INPUTS,
    board_reconstruction_n_inputs=TEST_INPUTS,
    analysis_on_cpu=True,
    precompute=PRECOMPUTE,
)


def test_full_chess_pipeline():

    autoencoder_group_paths = ["autoencoders/testing_chess"]
    csv_output_path = f"{autoencoder_group_paths[0]}/results.csv"
    final_output_path = f"{autoencoder_group_paths[0]}/f1_results.csv"

    full_pipeline.analyze_sae_groups(autoencoder_group_paths, csv_output_path, TEST_CONFIG)

    df = pd.read_csv(final_output_path)

    actual_result = df[
        "board_to_piece_masked_blank_and_initial_state_best_f1_score_per_class"
    ].max()

    expected_result = 0.7595600485801697

    assert actual_result - expected_result < TOLERANCE


def test_full_othello_pipeline():

    autoencoder_group_paths = ["autoencoders/testing_othello"]
    csv_output_path = f"{autoencoder_group_paths[0]}/results.csv"
    final_output_path = f"{autoencoder_group_paths[0]}/f1_results.csv"

    full_pipeline.analyze_sae_groups(autoencoder_group_paths, csv_output_path, TEST_CONFIG)

    df = pd.read_csv(final_output_path)

    actual_result = df[
        "games_batch_to_state_stack_mine_yours_blank_mask_BLRRC_best_f1_score_per_class"
    ].max()

    expected_result = 0.928957581520081

    assert actual_result - expected_result < TOLERANCE


def test_batched_chess_dataset_creation():

    single_custom_function = [chess_utils.board_to_piece_masked_blank_and_initial_state]
    single_func_dataset = eval_sae_as_classifier.construct_dataset(
        othello=False,
        custom_functions=single_custom_function,
        n_inputs=TEST_INPUTS,
        split="train",
        device=DEVICE,
        precompute_dataset=True,
    )

    all_custom_functions = TEST_CONFIG.chess_functions
    all_func_dataset = eval_sae_as_classifier.construct_dataset(
        othello=False,
        custom_functions=all_custom_functions,
        n_inputs=TEST_INPUTS,
        split="train",
        device=DEVICE,
        precompute_dataset=True,
    )

    assert torch.equal(
        single_func_dataset["board_to_piece_masked_blank_and_initial_state"],
        all_func_dataset["board_to_piece_masked_blank_and_initial_state"],
    )
