from nnsight import LanguageModel
import torch
import json
import pickle
from datasets import load_dataset

import circuits.chess_utils as chess_utils


def construct_eval_dataset(
    configs: list[chess_utils.Config], n_inputs: int, max_str_length: int = 256, device: str = "cpu"
):
    dataset = load_dataset("adamkarvonen/chess_sae_individual_games_filtered", streaming=False)
    pgn_strings = []
    for i, example in enumerate(dataset["train"]):
        if i >= n_inputs:
            break
        pgn_strings.append(example["text"][:max_str_length])

    data = {}
    data["pgn_strings"] = pgn_strings

    for config in configs:
        func_name = config.custom_board_state_function.__name__
        state_stack = chess_utils.create_state_stacks(
            pgn_strings, config.custom_board_state_function
        ).to(device)
        # state stack shape: "modes sample_size pgn_str_length rows cols"

        assert state_stack.shape[1] == len(pgn_strings)
        assert state_stack.shape[2] == max_str_length

        data[func_name] = chess_utils.state_stack_to_one_hot(
            1, config.num_rows, config.num_cols, config.min_val, config.max_val, device, state_stack
        )

        # data[func_name] shape "modes sample_size pgn_str_length rows cols num_options"

    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)

    return data
