import pickle
import pandas as pd
from typing import Callable
import torch
import os

import circuits.eval_sae_as_classifier as eval_sae
import circuits.analysis as analysis
import circuits.eval_board_reconstruction as eval_board_reconstruction
import circuits.get_eval_results as get_eval_results
import circuits.f1_analysis as f1_analysis
import circuits.chess_utils as chess_utils
import circuits.pipeline_config as p_config
import circuits.utils as utils

# For multi-GPU evaluation
from collections import deque
from joblib import Parallel, delayed


def initialize_dataframe(custom_functions: list[Callable]) -> pd.DataFrame:
    constant_columns = [
        "autoencoder_group_path",
        "autoencoder_path",
        "reconstruction_file",
        "trainer_class",
        "sae_class",
        "eval_sae_n_inputs",
        "eval_results_n_inputs",
        "board_reconstruction_n_inputs",
        "l0",
        "l1_loss",
        "l2_loss",
        "frac_alive",
        "frac_variance_explained",
        "cossim",
        "l2_ratio",
        "loss_original",
        "loss_reconstructed",
        "loss_zero",
        "frac_recovered",
        "num_alive_features",
    ]

    template_columns = [
        "board_reconstruction_board_count",
        "num_squares",
        "best_idx",
        "zero_L0",
        "zero_f1_score_per_class",
        "best_L0",
        "best_f1_score_per_class",
        "last_f1_score_per_class",
        "zero_num_true_positive_squares",
        "best_num_true_positive_squares",
        "zero_num_false_positive_squares",
        "best_num_false_positive_squares",
        "zero_num_false_negative_squares",
        "best_num_false_negative_squares",
        "zero_multiple_classes",
        "best_multiple_classes",
        "zero_num_true_and_false_positive_squares",
        "best_num_true_and_false_positive_squares",
        "high_precision_counts_per_T",
        "high_precision_and_recall_counts_per_T",
    ]

    # Generate the custom columns based on the custom functions
    custom_columns = [
        f"{func.__name__}_{template_col}"
        for func in custom_functions
        for template_col in template_columns
    ]

    # Combine the constant columns with the custom columns
    all_columns = constant_columns + custom_columns

    # Create and return the DataFrame with the combined columns
    return pd.DataFrame(columns=all_columns)


def append_results(
    eval_results: dict,
    aggregate_results: dict,
    board_reconstruction_results: dict,
    misc_stats: dict,
    custom_functions: list[Callable],
    df: pd.DataFrame,
    autoencoder_group_path: str,
    autoencoder_path: str,
    reconstruction_file: str,
) -> pd.DataFrame:
    # Initialize the new row with constant fields
    new_row = {
        "autoencoder_group_path": autoencoder_group_path,
        "autoencoder_path": autoencoder_path,
        "reconstruction_file": reconstruction_file,
        "trainer_class": aggregate_results["trainer_class"],
        "sae_class": aggregate_results["sae_class"],
        "eval_sae_n_inputs": aggregate_results["hyperparameters"]["n_inputs"],
        "eval_results_n_inputs": eval_results["hyperparameters"]["n_inputs"],
        "board_reconstruction_n_inputs": board_reconstruction_results["hyperparameters"][
            "n_inputs"
        ],
        "l0": eval_results["eval_results"]["l0"],
        "l1_loss": eval_results["eval_results"]["l1_loss"],
        "l2_loss": eval_results["eval_results"]["l2_loss"],
        "frac_alive": eval_results["eval_results"]["frac_alive"],
        "frac_variance_explained": eval_results["eval_results"]["frac_variance_explained"],
        "cossim": eval_results["eval_results"]["cossim"],
        "l2_ratio": eval_results["eval_results"]["l2_ratio"],
        "loss_original": eval_results["eval_results"]["loss_original"],
        "loss_reconstructed": eval_results["eval_results"]["loss_reconstructed"],
        "loss_zero": eval_results["eval_results"]["loss_zero"],
        "frac_recovered": eval_results["eval_results"]["frac_recovered"],
        "num_alive_features": board_reconstruction_results["alive_features"].shape[0],
    }

    for custom_function in custom_functions:
        function_name = custom_function.__name__
        best_idx = board_reconstruction_results[function_name]["f1_score_per_square"][:-1].argmax()
        last_idx = -1

        # Add the custom fields to the new row
        new_row[f"{function_name}_board_reconstruction_board_count"] = board_reconstruction_results[
            function_name
        ]["num_boards"]
        new_row[f"{function_name}_num_squares"] = board_reconstruction_results[function_name][
            "num_squares"
        ]
        new_row[f"{function_name}_best_idx"] = best_idx.item()
        new_row[f"{function_name}_zero_L0"] = board_reconstruction_results["active_per_token"][
            0
        ].item()
        new_row[f"{function_name}_best_L0"] = board_reconstruction_results["active_per_token"][
            best_idx
        ].item()
        new_row[f"{function_name}_zero_f1_score_per_class"] = board_reconstruction_results[
            function_name
        ]["f1_score_per_class"][0].item()
        new_row[f"{function_name}_best_f1_score_per_class"] = board_reconstruction_results[
            function_name
        ]["f1_score_per_class"][best_idx].item()
        new_row[f"{function_name}_last_f1_score_per_class"] = board_reconstruction_results[
            function_name
        ]["f1_score_per_class"][last_idx].item()
        new_row[f"{function_name}_zero_num_true_positive_squares"] = board_reconstruction_results[
            function_name
        ]["num_true_positive_squares"][0].item()
        new_row[f"{function_name}_best_num_true_positive_squares"] = board_reconstruction_results[
            function_name
        ]["num_true_positive_squares"][best_idx].item()
        new_row[f"{function_name}_zero_num_false_positive_squares"] = board_reconstruction_results[
            function_name
        ]["num_false_positive_squares"][0].item()
        new_row[f"{function_name}_best_num_false_positive_squares"] = board_reconstruction_results[
            function_name
        ]["num_false_positive_squares"][best_idx].item()
        new_row[f"{function_name}_zero_num_false_negative_squares"] = board_reconstruction_results[
            function_name
        ]["num_false_negative_squares"][0].item()
        new_row[f"{function_name}_best_num_false_negative_squares"] = board_reconstruction_results[
            function_name
        ]["num_false_negative_squares"][best_idx].item()
        new_row[f"{function_name}_zero_multiple_classes"] = board_reconstruction_results[
            function_name
        ]["num_multiple_classes"][0].item()
        new_row[f"{function_name}_best_multiple_classes"] = board_reconstruction_results[
            function_name
        ]["num_multiple_classes"][best_idx].item()
        new_row[f"{function_name}_zero_num_true_and_false_positive_squares"] = (
            board_reconstruction_results[
                function_name
            ]["num_true_and_false_positive_squares"][0].item()
        )
        new_row[f"{function_name}_best_num_true_and_false_positive_squares"] = (
            board_reconstruction_results[
                function_name
            ]["num_true_and_false_positive_squares"][best_idx].item()
        )
        new_row[f"{function_name}_high_precision_counts_per_T"] = misc_stats[function_name][
            "high_precision_counts_per_T"
        ]
        new_row[f"{function_name}_high_precision_and_recall_counts_per_T"] = misc_stats[
            function_name
        ]["high_precision_and_recall_counts_per_T"]

    new_row_df = pd.DataFrame([new_row])

    # Check if the original DataFrame is empty
    if df.empty:
        df = new_row_df
    else:
        df = pd.concat([df, new_row_df], ignore_index=True)
    return df


def check_all_sae_groups(autoencoder_group_paths: list[str]) -> bool:
    prev_othello = None
    for path in autoencoder_group_paths:
        assert os.path.isdir(path), f"Directory does not exist: {path}"
        cur_othello = eval_sae.check_if_autoencoder_is_othello(path)
        if prev_othello is not None:
            assert (
                cur_othello == prev_othello
            ), "All autoencoders in a group must be trained on the same game"
        prev_othello = cur_othello
    return cur_othello


def analyze_sae_groups(
    autoencoder_group_paths: list[str], csv_output_path: str, config: p_config.Config
):
    RESOURCE_STACK = deque([f"cuda:{i}" for i in range(config.N_GPUS)])

    othello = check_all_sae_groups(autoencoder_group_paths)

    dataset_size = max(config.eval_sae_n_inputs, config.board_reconstruction_n_inputs)

    # We have plenty of data and eval_results_data doesn't use VRAM, so we can afford to make it large
    # So we don't hit the end of the activation buffer
    eval_results_dataset_size = config.eval_results_n_inputs * 10

    indexing_functions = eval_sae.get_recommended_indexing_functions(othello)
    indexing_function = indexing_functions[0]

    if othello:
        custom_functions = config.othello_functions
        game_name = "othello"
    else:
        custom_functions = config.chess_functions
        game_name = "chess"

    train_dataset_name = f"{game_name}_train_dataset.pkl"
    test_dataset_name = f"{game_name}_test_dataset.pkl"

    device = RESOURCE_STACK.pop()

    if os.path.exists(train_dataset_name) and config.precompute:
        print("Loading statistics aggregation dataset")
        with open(train_dataset_name, "rb") as f:
            train_data = pickle.load(f)
    else:
        print("Constructing statistics aggregation dataset")
        train_data = eval_sae.construct_dataset(
            othello,
            custom_functions,
            dataset_size,
            split="train",
            device=device,
            precompute_dataset=config.precompute,
        )
        if config.precompute:
            print("Saving statistics aggregation dataset")
            with open(train_dataset_name, "wb") as f:
                pickle.dump(train_data, f)

    if os.path.exists(test_dataset_name) and config.precompute:
        print("Loading test dataset")
        with open(test_dataset_name, "rb") as f:
            test_data = pickle.load(f)
    else:
        print("Constructing test dataset")
        test_data = eval_sae.construct_dataset(
            othello,
            custom_functions,
            dataset_size,
            split="test",
            device=device,
            precompute_dataset=config.precompute,
        )
        if config.precompute:
            print("Saving test dataset")
            with open(test_dataset_name, "wb") as f:
                pickle.dump(test_data, f)

    eval_results_data = eval_sae.construct_dataset(
        othello,
        [],
        eval_results_dataset_size,
        split="train",
        device=device,
        precompute_dataset=config.precompute,
    )

    RESOURCE_STACK.append(device)
    del device

    for autoencoder_group_path in autoencoder_group_paths:
        new_othello = eval_sae.check_if_autoencoder_is_othello(autoencoder_group_path)
        assert (
            new_othello == othello
        ), "All autoencoders in a group must be trained on the same game"

        folders = eval_sae.get_nested_folders(autoencoder_group_path)

        def full_eval_pipeline(autoencoder_path):
            torch.cuda.empty_cache()

            df = initialize_dataframe(custom_functions)

            # For debugging
            # if "ef=4_lr=1e-03_l1=1e-01_layer_5" not in autoencoder_path:
            #     return df

            # Grab a GPU off the stack to use
            device = RESOURCE_STACK.pop()

            expected_eval_results_output_location = get_eval_results.get_output_location(
                autoencoder_path, n_inputs=config.eval_results_n_inputs
            )

            # In eval_results, we get SAE metrics like L0, loss recovered, etc. Takes minimal runtime
            if config.run_eval_results:
                # If this is set, everything below should be reproducible
                # Then we can just save results from 1 run, make optimizations, and check that the results are the same
                # The determinism is only needed for getting activations from the activation buffer for finding alive features
                torch.manual_seed(0)
                eval_results = get_eval_results.get_evals(
                    autoencoder_path,
                    config.eval_results_n_inputs,
                    config.batch_size,
                    device,
                    utils.to_device(eval_results_data.copy(), device),
                    othello=othello,
                    save_results=config.save_results,
                )
            else:
                with open(expected_eval_results_output_location, "rb") as f:
                    eval_results = pickle.load(f)
                eval_results = utils.to_device(eval_results, device)

            expected_aggregation_output_location = eval_sae.get_output_location(
                autoencoder_path,
                n_inputs=config.eval_sae_n_inputs,
                indexing_function=indexing_function,
            )

            # In eval_sae.aggreagate_statistics, we find the probability distribution over the board state over every feature
            # at every activation threshold. This is used to calculate future metrics like board reconstruction and coverage.
            if config.run_eval_sae:
                print("Aggregating", autoencoder_path)
                aggregation_results = eval_sae.aggregate_statistics(
                    custom_functions=custom_functions,
                    autoencoder_path=autoencoder_path,
                    n_inputs=config.eval_sae_n_inputs,
                    batch_size=config.batch_size,
                    device=device,
                    data=utils.to_device(train_data.copy(), device),
                    thresholds_T=config.f1_analysis_thresholds,
                    indexing_function=indexing_function,
                    othello=othello,
                    save_results=config.save_results,
                    precomputed=config.precompute,
                )
            else:
                with open(expected_aggregation_output_location, "rb") as f:
                    aggregation_results = pickle.load(f)
                aggregation_results = utils.to_device(aggregation_results, device)

            if config.analysis_on_cpu:
                aggregation_results = utils.to_device(aggregation_results, "cpu")
                analysis_device = "cpu"
            else:
                analysis_device = device

            # In analysis.analyze_results_dict, we "label" the features. Any board state that is present with
            # at least high_threshold probability is labeled as 1. This is only used for board reconstruction.
            expected_feature_labels_output_location = expected_aggregation_output_location.replace(
                "results.pkl", "feature_labels.pkl"
            )
            if config.run_analysis:
                feature_labels, misc_stats = analysis.analyze_results_dict(
                    aggregation_results,
                    output_path=expected_feature_labels_output_location,
                    device=analysis_device,
                    high_threshold=config.analysis_high_threshold,
                    low_threshold=config.analysis_low_threshold,
                    significance_threshold=config.analysis_significance_threshold,
                    verbose=False,
                    print_results=False,
                    save_results=config.save_feature_labels,
                )
            else:
                with open(expected_feature_labels_output_location, "rb") as f:
                    feature_labels = pickle.load(f)
                feature_labels = utils.to_device(feature_labels, device)

            if config.analysis_on_cpu:
                aggregation_results = utils.to_device(aggregation_results, device)
                feature_labels = utils.to_device(feature_labels, device)
                misc_stats = utils.to_device(misc_stats, device)

            expected_reconstruction_output_location = expected_aggregation_output_location.replace(
                "results.pkl", "reconstruction.pkl"
            )

            # In eval_board_reconstruction, we used our `feature_labels` to reconstruct the board state
            # on an unseen test dataset. We then can calculate the F1 score of the reconstruction.
            if config.run_board_reconstruction:
                print("Testing board reconstruction")
                board_reconstruction_results = eval_board_reconstruction.test_board_reconstructions(
                    custom_functions=custom_functions,
                    autoencoder_path=autoencoder_path,
                    feature_labels=feature_labels,
                    output_file=expected_reconstruction_output_location,
                    n_inputs=config.board_reconstruction_n_inputs,
                    batch_size=config.batch_size,
                    device=device,
                    data=utils.to_device(test_data.copy(), device),
                    othello=othello,
                    print_results=False,
                    save_results=config.save_results,
                    precomputed=config.precompute,
                )
            else:
                with open(expected_reconstruction_output_location, "rb") as f:
                    board_reconstruction_results = pickle.load(f)
                board_reconstruction_results = utils.to_device(board_reconstruction_results, device)

            # Add all the results to the dataframe for plotting purposes
            df = append_results(
                eval_results,
                aggregation_results,
                board_reconstruction_results,
                misc_stats,
                custom_functions,
                df,
                autoencoder_group_path,
                autoencoder_path,
                expected_reconstruction_output_location,
            )

            print("Finished", autoencoder_path)

            # Save the dataframe after each autoencoder so we don't lose data if the script crashes
            output_file = autoencoder_path + "/" + "results.csv"
            df.to_csv(output_file)

            # Put the GPU back on the stack after we're done
            RESOURCE_STACK.append(device)
            return df

        dfs = Parallel(n_jobs=config.N_GPUS, require="sharedmem")(
            delayed(full_eval_pipeline)(autoencoder_path) for autoencoder_path in folders
        )

        pd.concat(dfs, axis=0, ignore_index=True).to_csv(autoencoder_group_path + "results.csv")

    utils.concatenate_csv_files(
        file_list=[
            autoencoder_group_path + "results.csv"
            for autoencoder_group_path in autoencoder_group_paths
        ],
        output_file=csv_output_path,
    )

    results_filename_filter = str(config.eval_sae_n_inputs) + "_"
    f1_analysis_thresholds = config.f1_analysis_thresholds.tolist()

    RESOURCE_STACK = deque([f"cuda:{i}" for i in range(config.N_GPUS)])

    device = RESOURCE_STACK.pop()

    # Coverage is calculated here
    f1_analysis.add_coverage_to_df(
        autoencoder_group_paths,
        csv_output_path,
        results_filename_filter,
        device,
        f1_analysis_thresholds,
    )

    RESOURCE_STACK.append(device)
    del device


# NOTE: We are going to check that all autoencoders in a given group_path are for Chess XOR Othello
# However, we are not going to enfore that autoencoders in different group_paths are for the same game


# To reproduce paper results, download the desired SAEs using the directions in the `autoencoders/` directory.
# Then, uncomment the desired group_paths and output_path variables below.
# By default, at the bottom of this file, we have a test configuration that will run on the testing SAEs.

othello_group_paths = [
    "autoencoders/othello-trained_model-layer_5-2024-05-23/othello-trained_model-layer_5-gated",
    "autoencoders/othello-trained_model-layer_5-2024-05-23/othello-trained_model-layer_5-gated_anneal",
    "autoencoders/othello-trained_model-layer_5-2024-05-23/othello-trained_model-layer_5-p_anneal",
    "autoencoders/othello-trained_model-layer_5-2024-05-23/othello-trained_model-layer_5-standard",
]
othello_output_path = "autoencoders/othello-trained_model-layer_5-2024-05-23/results.csv"

# othello_random_group_paths = [
#     "autoencoders/othello-random_model-layer_5-standard",
# ]
# othello_random_output_path = "autoencoders/othello-random_model-layer_5-standard/results.csv"

chess_group_paths = [
    "autoencoders/chess-trained_model-layer_5-2024-05-23/chess-trained_model-layer_5-gated",
    "autoencoders/chess-trained_model-layer_5-2024-05-23/chess-trained_model-layer_5-gated_anneal",
    "autoencoders/chess-trained_model-layer_5-2024-05-23/chess-trained_model-layer_5-p_anneal",
    "autoencoders/chess-trained_model-layer_5-2024-05-23/chess-trained_model-layer_5-standard",
]
chess_output_path = "autoencoders/chess-trained_model-layer_5-2024-05-23/results.csv"

# chess_random_group_paths = [
#     "autoencoders/chess-random_model-layer_5-standard",
# ]
# chess_random_output_path = "autoencoders/chess-random_model-layer_5-standard/results.csv"

# othello_mlp_group_paths = ["autoencoders/othello_mlp_acts_identity_aes/"]
# othello_mlp_output_path = "autoencoders/othello_mlp_acts_identity_aes/results.csv"

# chess_mlp_group_paths = ["autoencoders/chess_mlp_acts_identity_aes/"]
# chess_mlp_output_path = "autoencoders/chess_mlp_acts_identity_aes/results.csv"

# othello_all_layers_group_paths = ["autoencoders/all_layers_othello_p_anneal_0530/"]
# othello_all_layers_output_path = "autoencoders/all_layers_othello_p_anneal_0530/results.csv"

# chess_all_layers_group_paths = ["autoencoders/chess_all_layers_resid/"]
# chess_all_layers_output_path = "autoencoders/chess_all_layers_resid/results.csv"

# othello_groups = [
#     (othello_group_paths, othello_output_path),
#     (othello_random_group_paths, othello_random_output_path),
#     (othello_mlp_group_paths, othello_mlp_output_path),
#     (othello_all_layers_group_paths, othello_all_layers_output_path),
# ]
# chess_groups = [
#     (chess_group_paths, chess_output_path),
#     (chess_random_group_paths, chess_random_output_path),
#     (chess_mlp_group_paths, chess_mlp_output_path),
#     (chess_all_layers_group_paths, chess_all_layers_output_path),
# ]

all_groups = [(chess_group_paths, chess_output_path), (othello_group_paths, othello_output_path)]

othello_test_path = ["autoencoders/testing_othello/"]
othello_test_output_path = "autoencoders/testing_othello/results.csv"

chess_test_path = ["autoencoders/testing_chess/"]
chess_test_output_path = "autoencoders/testing_chess/results.csv"

all_groups = [
    (chess_test_path, chess_test_output_path),
    (othello_test_path, othello_test_output_path),
]

# othello_groups = [(othello_test_path, othello_test_output_path)]
# chess_groups = [(chess_test_path, chess_test_output_path)]

if __name__ == "__main__":
    main_config = p_config.Config()

    # To edit the main_config, you can do things like:
    # main_config.eval_sae_n_inputs = 1000
    # main_config.N_GPUS = 1

    for group_path, output_path in all_groups:
        analyze_sae_groups(group_path, output_path, main_config)
