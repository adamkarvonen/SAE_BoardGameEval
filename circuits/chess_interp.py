from circuitsvis.activations import text_neuron_activations
from einops import rearrange
import torch
from tqdm import tqdm


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
