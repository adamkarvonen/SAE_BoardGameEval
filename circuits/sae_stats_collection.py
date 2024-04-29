import os
from nnsight import LanguageModel
import torch
import json
import pickle
import gc

from circuits.dictionary_learning import ActivationBuffer
from circuits.dictionary_learning.utils import hf_dataset_to_generator
from circuits.dictionary_learning import AutoEncoder
from circuits.dictionary_learning.evaluation import evaluate

from circuits.nanogpt_to_hf_transformers import NanogptTokenizer, convert_nanogpt_model
from circuits.chess_interp import examine_dimension_chess
import circuits.chess_utils as chess_utils
import circuits.chess_interp as chess_interp


def get_nested_folders(path: str) -> list[str]:
    """Get a list of folders nested one level deep in the given path."""
    folder_names = []
    # Process current directory and one level deep subdirectories
    for folder in os.listdir(path):
        if folder == "utils":
            continue
        current_folder = os.path.join(path, folder)
        if os.path.isdir(current_folder):
            if "ae.pt" in os.listdir(current_folder):
                folder_names.append(current_folder + "/")
            for subfolder in os.listdir(current_folder):  # Process subfolders
                subfolder_path = os.path.join(current_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    if "ae.pt" in os.listdir(subfolder_path):
                        folder_names.append(subfolder_path + "/")

    return folder_names


@torch.no_grad()
def get_feature(
    activations,
    ae: AutoEncoder,
    device,
) -> torch.Tensor:
    try:
        x = next(activations).to(device)
    except StopIteration:
        raise StopIteration(
            "Not enough activations in buffer. Pass a buffer with a smaller batch size or more data."
        )

    x_hat, f = ae(x, output_features=True)

    return f


def get_ae_stats(
    autoencoder_path: str,
    max_dims: int,
    n_inputs: int,
    top_k: int,
    batch_size: int,
    device: str,
    model_path: str = "models/",
    save_results: bool = False,
) -> tuple[dict, dict]:

    autoencoder_model_path = f"{autoencoder_path}ae.pt"
    autoencoder_config_path = f"{autoencoder_path}config.json"
    ae = AutoEncoder.from_pretrained(autoencoder_model_path, device=device)

    with open(autoencoder_config_path, "r") as f:
        config = json.load(f)

    context_length = config["buffer"]["ctx_len"]
    layer = config["trainer"]["layer"]

    tokenizer = NanogptTokenizer(meta_path=f"{model_path}meta.pkl")
    model = convert_nanogpt_model(
        f"{model_path}lichess_8layers_ckpt_no_optimizer.pt", torch.device(device)
    )
    model = LanguageModel(model, device_map=device, tokenizer=tokenizer).to(device)

    submodule = model.transformer.h[layer].mlp  # layer 1 MLP
    activation_dim = config["trainer"]["activation_dim"]  # output dimension of the MLP
    dictionary_size = config["trainer"]["dictionary_size"]

    # chess_sae_test is 100MB of data, so no big deal to download it
    data = hf_dataset_to_generator("adamkarvonen/chess_sae_test", streaming=False)
    buffer = ActivationBuffer(
        data,
        model,
        submodule,
        n_ctxs=512,
        ctx_len=context_length,
        refresh_batch_size=batch_size,
        io="out",
        d_submodule=activation_dim,
        device=device,
        out_batch_size=batch_size,
    )

    total_inputs = 8000
    assert total_inputs % batch_size == 0
    num_iters = total_inputs // batch_size

    features = torch.zeros((total_inputs, dictionary_size), device=device)
    for i in range(num_iters):
        feature = get_feature(buffer, ae, device)  # (batch_size, dictionary_size)
        features[i * batch_size : (i + 1) * batch_size, :] = feature

    firing_rate_per_feature = (features != 0).float().sum(dim=0) / total_inputs

    assert firing_rate_per_feature.shape[0] == dictionary_size

    mask = (firing_rate_per_feature > 0) & (firing_rate_per_feature < 0.5)
    idx = torch.nonzero(mask, as_tuple=False).squeeze()

    per_dim_stats = examine_dimension_chess(
        model,
        submodule,
        buffer,
        dictionary=ae,
        dims=idx[:max_dims],
        n_inputs=n_inputs,
        k=top_k + 1,
        batch_size=25,
        processing_device=torch.device("cpu"),
    )

    eval_results = evaluate(
        ae, buffer, max_len=context_length, batch_size=batch_size, io="out", device=device
    )

    if save_results:
        pickle.dump(per_dim_stats, open(f"{autoencoder_path}per_dim_stats.pkl", "wb"))

    return per_dim_stats, eval_results


def compute_all_ae_stats(folder: str, save_results: bool = False):

    # TODO These should be passed as arguments or read from a config file
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_inputs = 5000
    top_k = 30
    max_dims = 4000
    batch_size = 25

    syntax_metrics = [
        chess_utils.find_num_indices,
        chess_utils.find_spaces_indices,
        chess_utils.find_dots_indices,
    ]
    board_metrics = [
        chess_utils.piece_config,
        chess_utils.threat_config,
        chess_utils.check_config,
        chess_utils.pin_config,
    ]

    folders = get_nested_folders(folder)

    print(f"Found {len(folders)} folders.")

    total_results = {}
    for folder in folders:
        print(f"Starting {folder}")
        per_dim_stats, eval_results = get_ae_stats(
            folder,
            max_dims=max_dims,
            n_inputs=n_inputs,
            top_k=top_k,
            batch_size=batch_size,
            device=device,
            model_path="models/",
            save_results=save_results,
        )
        results = {}
        results["syntax"] = {}
        results["eval_results"] = eval_results

        for metric in syntax_metrics:
            metric_name = metric.__name__
            results["syntax"][metric_name] = chess_interp.syntax_analysis(
                per_dim_stats, top_k, top_k, max_dims, metric
            )
        results["board"] = chess_interp.board_analysis(
            per_dim_stats, top_k, top_k, max_dims, 0.99, board_metrics
        )
        total_results[folder] = results

        print(f"Finished {folder}")

        del per_dim_stats
        del eval_results
        gc.collect()

    total_results = chess_interp.serialize_results(total_results)
    total_results["hyperparameters"] = {}
    total_results["hyperparameters"]["n_inputs"] = n_inputs
    total_results["hyperparameters"]["top_k"] = top_k
    total_results["hyperparameters"]["max_dims"] = max_dims
    json.dump(total_results, open(f"total_results.json", "w"))


if __name__ == "__main__":
    compute_all_ae_stats("autoencoders/")
