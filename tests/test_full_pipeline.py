import pandas as pd

import circuits.full_pipeline as full_pipeline
import circuits.pipeline_config as pipeline_config


def test_full_pipeline():

    test_config = pipeline_config.Config(
        eval_sae_n_inputs=25,
        batch_size=5,
        eval_results_n_inputs=25,
        board_reconstruction_n_inputs=25,
        analysis_on_cpu=True,
    )
    autoencoder_group_paths = ["autoencoders/testing_chess"]
    csv_output_path = "autoencoders/testing_chess/results.csv"
    final_output_path = "autoencoders/testing_chess/f1_results.csv"

    full_pipeline.analyze_sae_groups(autoencoder_group_paths, csv_output_path, test_config)

    df = pd.read_csv(final_output_path)

    actual_result = df[
        "board_to_piece_masked_blank_and_initial_state_best_f1_score_per_class"
    ].max()

    expected_result = 0.7595600485801697

    assert actual_result - expected_result < 1e-6
