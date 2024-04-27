from circuitsvis.activations import text_neuron_activations
from einops import rearrange
import torch
from tqdm import tqdm
import chess_utils
from chess_utils import Config

from pydantic import BaseModel


class BoardResultsConfig(BaseModel):
    dim_count: int = 0
    nonzero_count: int = 0
    pattern_match_count: int = 0
    total_average_length: float = 0.0
    average_matches_per_dim: float = 0.0
    per_class_dict: dict[int, int]
    board_tracker: list[list[int]]  # shape: (num_rows, num_cols)


class SyntaxResultsConfig(BaseModel):
    dim_count: int = 0
    nonzero_count: int = 0
    syntax_match_idx_count: int = 0
    average_input_length: float = 0.0


def serialize_results(data):
    if isinstance(data, BaseModel):
        return data.model_dump()
    elif isinstance(data, dict):
        return {key: serialize_results(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [serialize_results(item) for item in data]
    else:
        return data


@torch.no_grad()
def examine_dimension_chess(
    model,
    submodule,
    buffer,
    dictionary=None,
    max_length=128,
    n_inputs=512,
    dims=torch.tensor([0]),
    k=30,
    batch_size=4,
    processing_device=torch.device("cpu"),
):
    """I have made the following modifications:
    - We can now pass in a tensor of dimensions to examine, rather than just a single dimension.
    - We iterate over inputs by step batch_size to reduce VRAM usage.
    - We now return a dictionary of namedtuples, where the keys are the dimensions and the values are namedtuples
      containing the top contexts, top tokens, top affected tokens, decoded tokens, and activations for that dimension.
    - top_contexts is None unless len(dims) == 1.
    - I'm currently not returning the top affected tokens for now.
    Much more efficient processing (50x faster) for large dim counts.
    - Processing_device of cpu vs cuda doesn't make much runtime difference, but lowers VRAM usage.
    """

    assert n_inputs % batch_size == 0
    n_iters = n_inputs // batch_size

    dim_count = dims.shape[0]

    activations = torch.zeros(
        (dim_count, n_inputs, max_length), device=processing_device, dtype=torch.float32
    )
    tokens = torch.zeros((n_inputs, max_length), device=processing_device, dtype=torch.int64)

    for i in tqdm(range(n_iters), total=n_iters, desc="Collecting activations"):
        inputs = buffer.text_batch(batch_size=batch_size)
        with model.trace(inputs, invoker_args=dict(max_length=max_length, truncation=True)):
            cur_tokens = model.input[1][
                "input_ids"
            ].save()  # if you're getting errors, check here; might only work for pythia models
            cur_activations = submodule.output
            if type(cur_activations.shape) == tuple:
                cur_activations = cur_activations[0]
            if dictionary is not None:
                cur_activations = dictionary.encode(cur_activations)
            cur_activations = cur_activations[
                :, :, dims
            ].save()  # Shape: (batch_size, max_length, dim_count)
        cur_activations = rearrange(
            cur_activations.value, "b n d -> d b n"
        )  # Shape: (dim_count, batch_size, max_length)
        activations[:, i * batch_size : (i + 1) * batch_size, :] = cur_activations
        tokens[i * batch_size : (i + 1) * batch_size, :] = cur_tokens.value

    activations = activations.to("cpu")
    tokens = tokens.to("cpu")
    decoded_tokens = [model.tokenizer.decode(tokens[i]) for i in range(tokens.shape[0])]

    per_dim_stats = {}
    idxs_dict = {}
    vocab_size = 32

    for i in range(vocab_size):
        idxs_dict[i] = (tokens == i).nonzero(as_tuple=True)

    for i, dim in tqdm(enumerate(dims), total=len(dims), desc="Processing activations"):
        individual_acts = activations[i]

        # top_affected = feature_effect(model, submodule, dictionary, dim_idx, tokens, k=k)
        # top_affected = [(model.tokenizer.decode(tok), prob.item()) for tok, prob in zip(*top_affected)]
        top_affected = None  # Uses too much compute for large dims

        # get top k tokens by mean activation
        token_mean_acts = {}
        for tok in idxs_dict:
            idxs = idxs_dict[tok]
            token_mean_acts[tok] = individual_acts[idxs].mean().item()
        top_tokens = sorted(token_mean_acts.items(), key=lambda x: x[1], reverse=True)[:k]
        top_tokens = [(model.tokenizer.decode(tok), act) for tok, act in top_tokens]

        flattened_acts = rearrange(individual_acts, "b n -> (b n)")
        topk_indices = torch.argsort(flattened_acts, dim=0, descending=True)[:k]
        batch_indices = topk_indices // individual_acts.shape[1]
        token_indices = topk_indices % individual_acts.shape[1]

        # .clone() is necessary for saving results with pickle. Otherwise, everything is saved as a reference to the same tensor.
        individual_acts = [
            individual_acts[batch_idx, : token_id + 1, None, None].clone()
            for batch_idx, token_id in zip(batch_indices, token_indices)
        ]
        individual_tokens = [
            decoded_tokens[batch_idx][: token_idx + 1]
            for batch_idx, token_idx in zip(batch_indices, token_indices)
        ]

        if dim_count == 1:
            top_contexts = text_neuron_activations(individual_tokens, activations)
        else:
            top_contexts = None

        dim_stats = {}
        dim_stats["top_contexts"] = top_contexts
        dim_stats["top_tokens"] = top_tokens
        dim_stats["top_affected"] = top_affected
        dim_stats["decoded_tokens"] = individual_tokens
        dim_stats["activations"] = individual_acts

        per_dim_stats[dim.item()] = dim_stats

    return per_dim_stats


def syntax_analysis(
    per_dim_stats: dict,
    minimum_number_of_activations: int,
    top_k: int,
    max_dims: int,
    syntax_function: callable,
    verbose: bool = False,
) -> SyntaxResultsConfig:

    results = SyntaxResultsConfig()

    for dim in per_dim_stats:
        results.dim_count += 1
        if results.dim_count > max_dims:
            break

        decoded_tokens = per_dim_stats[dim]["decoded_tokens"]
        activations = per_dim_stats[dim]["activations"]
        # If the dim doesn't have at least 10 firing activations, skip it
        if activations[minimum_number_of_activations][-1].item() == 0:
            continue
        results.nonzero_count += 1

        inputs = ["".join(string) for string in decoded_tokens]
        inputs = inputs[:top_k]

        num_indices = []
        count = 0
        for i, pgn in enumerate(inputs[:top_k]):
            # NOTE: Uncomment this line to view examples of common indices
            # print(f"dim: {dim} pgn: {pgn}, activation: {activations[i][-1].item()}")
            nums = syntax_function(pgn)
            num_indices.append(nums)

            # If the last token (which contains the max activation for that context) is a number
            # Then we count this firing as a "number index firing"
            if (len(pgn) - 1) in nums:
                count += 1

        if count == top_k:
            # print(f"All top {top_k} activations in dim: {dim} are on num indices")
            results.syntax_match_idx_count += 1
            average_input_length = sum(len(pgn) for pgn in inputs[:top_k]) / len(inputs[:top_k])
            results.average_input_length += average_input_length

    if results.syntax_match_idx_count > 0:
        results.average_input_length /= results.syntax_match_idx_count

    if verbose:
        print(
            f"Out of {len(per_dim_stats)} features, {results.nonzero_count} had at least {minimum_number_of_activations} activations."
        )
        print(
            f"{results.syntax_match_idx_count} features matched on all top {top_k} inputs for our syntax function {syntax_function.__name__}"
        )
        print(
            f"The average length of inputs of pattern matching features was {results.average_input_length:.2f}"
        )

    return results


def board_analysis(
    per_dim_stats: dict,
    minimum_number_of_activations: int,
    top_k: int,
    max_dims: int,
    threshold: float,
    configs: list[Config],
    device: str = "cpu",
    verbose: bool = False,
) -> dict[str, BoardResultsConfig]:

    nonzero_count = 0
    dim_count = 0

    results: dict[str, BoardResultsConfig] = {}

    for config in configs:
        board_tracker = torch.zeros(config.num_rows, config.num_cols).tolist()
        num_classes = abs(config.min_val) + abs(config.max_val) + 1
        per_class_dict = {key: 0 for key in range(0, num_classes)}

        results[config.custom_board_state_function.__name__] = BoardResultsConfig(
            per_class_dict=per_class_dict,
            board_tracker=board_tracker,
        )

    for dim in tqdm(per_dim_stats, total=len(per_dim_stats), desc="Processing chess pgn strings"):
        dim_count += 1
        if dim_count > max_dims:
            break

        decoded_tokens = per_dim_stats[dim]["decoded_tokens"]
        activations = per_dim_stats[dim]["activations"]
        # If the dim doesn't have at least minimum_number_of_activations firing activations, skip it
        if activations[minimum_number_of_activations][-1].item() == 0:
            continue
        nonzero_count += 1

        inputs = ["".join(string) for string in decoded_tokens]
        inputs = inputs[:top_k]

        count = 0

        chess_boards = [
            chess_utils.pgn_string_to_board(pgn, allow_exception=True) for pgn in inputs
        ]

        for config in configs:

            config_name = config.custom_board_state_function.__name__

            one_hot_list = chess_utils.chess_boards_to_state_stack(chess_boards, device, config)
            one_hot_list = chess_utils.mask_initial_board_states(one_hot_list, device, config)
            averaged_one_hot = chess_utils.get_averaged_states(one_hot_list)
            common_indices = chess_utils.find_common_states(averaged_one_hot, threshold)

            if any(len(idx) > 0 for idx in common_indices):  # if at least one square matches
                results[config_name].pattern_match_count += 1
                average_input_length = sum(len(pgn) for pgn in inputs) / len(inputs)
                results[config_name].total_average_length += average_input_length

            if config.num_rows == 8:
                for idx in zip(*common_indices):
                    value = averaged_one_hot[idx].item()
                    # print(f"Dim: {dim}, Average input length: {int(average_input_length):04}, Value: {value:.2f} at Index: {idx}")
                    results[config_name].board_tracker[idx[0]][idx[1]] += 1
                    results[config_name].per_class_dict[idx[2].item()] += 1
                    results[config_name].average_matches_per_dim += 1

    for config in configs:
        config_name = config.custom_board_state_function.__name__
        match_count = results[config_name].pattern_match_count
        results[config_name].dim_count = dim_count
        results[config_name].nonzero_count = nonzero_count
        results[config_name].board_tracker = results[config_name].board_tracker
        if match_count > 0:
            results[config_name].total_average_length /= match_count
            results[config_name].average_matches_per_dim /= match_count

    if verbose:
        for config in configs:
            config_name = config.custom_board_state_function.__name__
            pattern_match_count = results[config_name].pattern_match_count
            total_average_length = results[config_name].total_average_length
            print(f"\n{config_name} Results:")
            print(
                f"Out of {dim_count} features, {nonzero_count} had at least {minimum_number_of_activations} activations."
            )
            print(
                f"{pattern_match_count} features matched on all top {top_k} inputs for our board to state function {config_name}"
            )
            print(
                f"The average length of inputs of pattern matching features was {total_average_length}"
            )

            if config.num_rows == 8:
                board_tracker = results[config_name].board_tracker
                print(f"\nThe following square states had the following number of occurances:")
                for key, count in results[config_name].per_class_dict.items():
                    print(f"Index: {key}, Count: {count}")

                print(f"\nHere are the most common squares:")
                board_tracker = torch.tensor(board_tracker).flip(
                    0
                )  # torch.tensor has a cleaner printout
                print(board_tracker)

    return results