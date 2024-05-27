import torch
import pickle
import einops
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.colors import Normalize
from typing import Callable
import json

import circuits.analysis as analysis
import circuits.eval_sae_as_classifier as eval_sae
import circuits.chess_utils as chess_utils
import circuits.utils as utils


def mask_all_blanks(results: dict, device) -> dict:
    custom_functions = analysis.get_all_custom_functions(results)
    for function in custom_functions:
        function_name = function.__name__

        if (
            function == chess_utils.board_to_piece_state
            or function == chess_utils.board_to_piece_color_state
        ):
            on_TFRRC = results[function_name]["on"]
            off_TFRRC = results[function_name]["off"]
            results[function_name]["on"] = analysis.mask_initial_board_state(
                on_TFRRC, function, device
            )
            results[function_name]["off"] = analysis.mask_initial_board_state(
                off_TFRRC, function, device
            )

    return results


def best_f1_average(f1_TFRRC: torch.Tensor, config: chess_utils.Config) -> torch.Tensor:
    """For every threshold, for every square, find the best F1 score across all features. Then average across all squares.
    NOTE: If the function is binary, num_squares == 1. If it is board to piece state, num_squares == 8 * 8 * 12
    """
    f1_TRRC, _ = torch.max(f1_TFRRC, dim=1)

    T, R1, R2, C = f1_TRRC.shape

    if config.one_hot_mask_idx is not None:
        C -= 1

    max_possible = R1 * R2 * C

    f1_T = einops.reduce(f1_TRRC, "T R1 R2 C -> T", "sum") / max_possible

    return f1_T


def f1s_above_threshold(
    f1_TFRRC: torch.Tensor, threshold: float, config: chess_utils.Config
) -> tuple[torch.Tensor, torch.Tensor]:
    """For every threshold, for every square, find the best F1 score across all features. Then, find the number of squares that have a F1 score above the threshold.
    If the function is binary, num_squares == 1. If it is board to piece state, num_squares == 8 * 8 * 12
    NOTE: This will probably be most useful for features with 8x8xn options."""
    f1_TRRC, _ = torch.max(f1_TFRRC, dim=1)

    f1s_above_threshold_TRCC = f1_TRRC > threshold

    T, R1, R2, C = f1_TRRC.shape
    if config.one_hot_mask_idx is not None:
        C -= 1

    max_possible = R1 * R2 * C

    f1_T = einops.reduce(f1s_above_threshold_TRCC, "T R1 R2 C -> T", "sum")

    f1_T_normalized = f1_T / max_possible

    return f1_T, f1_T_normalized


def get_custom_functions(
    autoencoder_group_path: str, results_filename_filter: str, device: str
) -> list[Callable]:
    folders = eval_sae.get_nested_folders(autoencoder_group_path)
    first_autoencoder_path = folders[0]
    results_filenames = analysis.get_all_results_file_names(
        first_autoencoder_path, results_filename_filter
    )

    print(results_filenames)

    if len(results_filenames) > 1:
        raise ValueError("There are multiple results files")
    if len(results_filenames) == 0:
        raise ValueError("There are no results files")
    results_filename = results_filenames[0]

    with open(first_autoencoder_path + results_filename, "rb") as f:
        results = pickle.load(f)

    results = utils.to_device(results, device)

    custom_functions = analysis.get_all_custom_functions(results)
    return custom_functions


def get_custom_function_names(custom_functions: list[Callable]) -> list[str]:
    custom_function_names = [function.__name__ for function in custom_functions]
    return custom_function_names


def get_threshold_column_names(func_name: str, threshold: float) -> tuple[str, str]:
    return (
        f"{func_name}_f1_threshold_{threshold}",
        f"{func_name}_f1_threshold_{threshold}_normalized",
    )


def check_df_if_othello(df: pd.DataFrame) -> bool:
    for column in df.columns:
        if "games_batch_to_state_stack" in column:
            return True
        if "board_to_piece_" in column:
            return False

    raise ValueError("Could not determine if this is an Othello dataframe")


def get_all_sae_f1_results(
    autoencoder_group_paths: list[str],
    df: pd.DataFrame,
    results_filename_filter: str,
    custom_functions: list[Callable],
    custom_function_names: list[str],
    device: str,
    thresholds: list[float],
    mask: bool,
) -> dict:
    all_sae_results = {}

    for autoencoder_group_path in autoencoder_group_paths:

        folders = eval_sae.get_nested_folders(autoencoder_group_path)
        sae_results = {}

        for autoencoder_path in folders:

            print(f"Processing {autoencoder_path}")

            assert (
                autoencoder_path in df["autoencoder_path"].values
            ), f"{autoencoder_path} not in csv file"

            sae_results[autoencoder_path] = {}

            results_filenames = analysis.get_all_results_file_names(
                autoencoder_path, results_filename_filter
            )
            if len(results_filenames) > 1 or len(results_filenames) == 0:
                print(
                    f"Skipping {autoencoder_path} because it has {len(results_filenames)} results files"
                )
                print("This is most likely because there are results files from different n_inputs")
                continue
            results_filename = results_filenames[0]

            with open(autoencoder_path + results_filename, "rb") as f:
                results = pickle.load(f)

            results = utils.to_device(results, device)

            results = analysis.add_off_tracker(results, custom_functions, device)
            f1_dict_TFRRC = analysis.get_all_f1s(results, device)

            # feature_labels = analysis.analyze_results_dict(
            #     results,
            #     output_path="",
            #     device=device,
            #     high_threshold=0.95,
            #     low_threshold=0.1,
            #     significance_threshold=10,
            #     save_results=False,
            #     mask=mask,
            #     verbose=False,
            #     print_results=False,
            # )

            correct_row = df["autoencoder_path"] == autoencoder_path
            sae_results[autoencoder_path]["l0"] = df[correct_row]["l0"].values[0]
            sae_results[autoencoder_path]["frac_variance_explained"] = df[correct_row][
                "frac_variance_explained"
            ].values[0]

            for func_name in f1_dict_TFRRC:
                config = chess_utils.config_lookup[func_name]
                custom_function = config.custom_board_state_function
                assert (
                    custom_function in custom_functions
                ), f"Key {custom_function} not in custom_functions"
                f1_TFRRC = f1_dict_TFRRC[func_name]

                average_f1_T = best_f1_average(f1_TFRRC, config)
                sae_results[autoencoder_path][f"{func_name}_average_f1"] = average_f1_T

                for threshold in thresholds:
                    threshold_column, threshold_column_normalized = get_threshold_column_names(
                        func_name, threshold
                    )
                    f1_T, f1_T_normalized = f1s_above_threshold(f1_TFRRC, threshold, config)
                    sae_results[autoencoder_path][threshold_column] = f1_T
                    sae_results[autoencoder_path][threshold_column_normalized] = f1_T_normalized

            # torch.cuda.empty_cache()
        all_sae_results[autoencoder_group_path] = sae_results
    return all_sae_results


def update_dataframe_with_results(
    df: pd.DataFrame,
    all_sae_results: dict,
    custom_function_names: list[str],
    autoencoder_group_paths: list[str],
    thresholds: list[float],
) -> pd.DataFrame:
    assert df["autoencoder_path"].is_unique
    updates = []
    for autoencoder_group_path in autoencoder_group_paths:
        folders = eval_sae.get_nested_folders(autoencoder_group_path)
        for autoencoder_path in folders:
            results = {"autoencoder_path": autoencoder_path}
            for func_name in custom_function_names:

                f1_T = all_sae_results[autoencoder_group_path][autoencoder_path][
                    f"{func_name}_average_f1"
                ]
                best_idx = torch.argmax(f1_T)
                best_f1 = f1_T[best_idx]

                results[f"{func_name}_best_average_f1"] = best_f1.item()
                results[f"{func_name}_best_average_f1_idx"] = best_idx.item()
                results[f"{func_name}_all_average_f1s"] = json.dumps(f1_T.tolist())

                for threshold in thresholds:
                    threshold_column, threshold_normalized_column = get_threshold_column_names(
                        func_name, threshold
                    )
                    f1_T = all_sae_results[autoencoder_group_path][autoencoder_path][
                        threshold_column
                    ]
                    f1_T_normalized = all_sae_results[autoencoder_group_path][autoencoder_path][
                        threshold_normalized_column
                    ]
                    best_idx = torch.argmax(f1_T)
                    best_f1_at_threshold = f1_T[best_idx]
                    best_f1_normalized = f1_T_normalized[best_idx]

                    results[f"{func_name}_f1_threshold_{threshold}_best"] = (
                        best_f1_at_threshold.item()
                    )
                    results[f"{func_name}_f1_threshold_{threshold}_best_normalized"] = (
                        best_f1_normalized.item()
                    )
                    results[f"{func_name}_f1_threshold_{threshold}_best_idx"] = best_idx.item()
                    results[f"{func_name}_f1_threshold_{threshold}_best_normalized_idx"] = (
                        best_idx.item()
                    )
                    results[f"{func_name}_f1_threshold_{threshold}_all"] = json.dumps(f1_T.tolist())
                    results[f"{func_name}_f1_threshold_{threshold}_all_normalized"] = json.dumps(
                        f1_T_normalized.tolist()
                    )

            updates.append(results)

    update_df = pd.DataFrame(updates)
    df = pd.merge(df, update_df, on="autoencoder_path", how="outer")
    assert df["autoencoder_path"].is_unique
    return df


def add_average_board_reconstruction_for_columns(
    df: pd.DataFrame,
    average_metric_columns: list[str],
    average_metric_idx_columns: list[str],
    filter_columns: list[tuple[str, float]],
    BSP_type: str,
    metric_type: str = "best_f1_score_per_class",
) -> tuple[pd.DataFrame, list[str], list[str]]:

    combined_metric_name = f"{BSP_type}{metric_type}"
    epsilon = 1e-8

    filter_columns = [col for col, weight in filter_columns]

    true_positive_columns = []
    false_positive_columns = []
    false_negative_columns = []
    best_idx_columns = []

    for col in df.columns:
        if "best_num_true_positive" in col:
            if any([low_level in col for low_level in filter_columns]):
                true_positive_columns.append(col)
        if "best_num_false_positive" in col:
            if any([low_level in col for low_level in filter_columns]):
                false_positive_columns.append(col)
        if "best_num_false_negative" in col:
            if any([low_level in col for low_level in filter_columns]):
                false_negative_columns.append(col)

        if "best_idx" in col and "threshold" not in col:
            if any([low_level in col for low_level in filter_columns]):
                best_idx_columns.append(col)

    if len(true_positive_columns) == 0:
        raise ValueError("No true positive columns found")
    if len(false_positive_columns) == 0:
        raise ValueError("No false positive columns found")
    if len(false_negative_columns) == 0:
        raise ValueError("No false negative columns found")
    if len(best_idx_columns) == 0:
        raise ValueError("No best idx columns found")

    average_metric_column_name = f"average_{combined_metric_name}_f1"
    average_metric_idx_column_name = f"average_{combined_metric_name}_best_idx"

    df[f"average_{combined_metric_name}_true_positive"] = df[true_positive_columns].mean(axis=1)
    df[f"average_{combined_metric_name}_false_positive"] = df[false_positive_columns].mean(axis=1)
    df[f"average_{combined_metric_name}_false_negative"] = df[false_negative_columns].mean(axis=1)
    df[average_metric_idx_column_name] = df[best_idx_columns].mean(axis=1)

    df[f"average_{combined_metric_name}_precision"] = df[
        f"average_{combined_metric_name}_true_positive"
    ] / (
        df[f"average_{combined_metric_name}_true_positive"]
        + df[f"average_{combined_metric_name}_false_positive"]
        + epsilon
    )
    df[f"average_{combined_metric_name}_recall"] = df[
        f"average_{combined_metric_name}_true_positive"
    ] / (
        df[f"average_{combined_metric_name}_true_positive"]
        + df[f"average_{combined_metric_name}_false_negative"]
        + epsilon
    )
    df[average_metric_column_name] = (
        2
        * (
            df[f"average_{combined_metric_name}_precision"]
            * df[f"average_{combined_metric_name}_recall"]
        )
        / (
            df[f"average_{combined_metric_name}_precision"]
            + df[f"average_{combined_metric_name}_recall"]
            + epsilon
        )
    )

    average_metric_columns.append(average_metric_column_name)
    average_metric_idx_columns.append(average_metric_idx_column_name)

    # df[f'average_{metric_type}_f1'].fillna(0, inplace=True)  # Handling any NaN results

    return df, average_metric_columns, average_metric_idx_columns


def add_average_coverage_for_columns(
    df: pd.DataFrame,
    average_metric_columns: list[str],
    average_metric_idx_columns: list[str],
    filter_columns: list[tuple[str, float]],
    BSP_type: str,
    metric_type: str = "best_average_f1",
) -> tuple[pd.DataFrame, list[str], list[str]]:

    combined_metric_name = f"{BSP_type}{metric_type}"

    average_metric_column_name = f"average_{combined_metric_name}"
    average_metric_idx_column_name = f"average_{combined_metric_name}_best_idx"

    filter_columns_idx = []

    for column_name, weight in filter_columns:
        filter_columns_idx.append(f"{column_name}_best_average_f1_idx")

    df[average_metric_idx_column_name] = df[filter_columns_idx].mean(axis=1)

    df[average_metric_column_name] = 0
    total_weight = 0
    for column_name, weight in filter_columns:
        df[average_metric_column_name] += df[f"{column_name}_{metric_type}"] * weight
        total_weight += weight
    df[average_metric_column_name] /= total_weight

    average_metric_columns.append(average_metric_column_name)
    average_metric_idx_columns.append(average_metric_idx_column_name)

    return df, average_metric_columns, average_metric_idx_columns


def complete_analysis_pipeline(
    autoencoder_group_paths: list[str],
    csv_results_path: str,
    results_filename_filter: str,
    device: str,
    thresholds: list[float],
) -> str:
    mask = False
    df = pd.read_csv(csv_results_path)

    custom_functions = get_custom_functions(
        autoencoder_group_paths[0], results_filename_filter, device
    )

    custom_function_names = get_custom_function_names(custom_functions)
    all_sae_results = get_all_sae_f1_results(
        autoencoder_group_paths,
        df,
        results_filename_filter,
        custom_functions,
        custom_function_names,
        device,
        thresholds,
        mask,
    )
    df = update_dataframe_with_results(
        df, all_sae_results, custom_function_names, autoencoder_group_paths, thresholds
    )
    output_path = csv_results_path.replace("results.csv", "f1_results.csv")
    df.to_csv(output_path, index=False)
    return output_path
