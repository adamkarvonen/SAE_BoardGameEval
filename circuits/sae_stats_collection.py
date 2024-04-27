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

DEVICE = torch.device("cuda")
MODEL_PATH = "models/lichess_8layers_ckpt_no_optimizer.pt"
BATCH_SIZE = 8
TOP_K = 30
MAX_DIMS = 10000
N_INPUTS = 5000


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


def get_ae_stats(autoencoder_path: str, save_results: bool = False) -> tuple[dict, dict]:

    autoencoder_model_path = f"{autoencoder_path}ae.pt"
    autoencoder_config_path = f"{autoencoder_path}config.json"
    ae = AutoEncoder.from_pretrained(autoencoder_model_path, device=DEVICE)

    with open(autoencoder_config_path, "r") as f:
        config = json.load(f)

    context_length = config["buffer"]["ctx_len"]
    layer = config["trainer"]["layer"]

    tokenizer = NanogptTokenizer(meta_path="models/meta.pkl")
    model = convert_nanogpt_model(MODEL_PATH, torch.device(DEVICE))
    model = LanguageModel(model, device_map=DEVICE, tokenizer=tokenizer).to(DEVICE)

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
        refresh_batch_size=BATCH_SIZE,
        io="out",
        d_submodule=activation_dim,
        device=DEVICE,
        out_batch_size=BATCH_SIZE,
    )

    total_inputs = 8192
    assert total_inputs % BATCH_SIZE == 0
    num_iters = total_inputs // BATCH_SIZE

    features = torch.zeros((total_inputs, dictionary_size), device=DEVICE)
    for i in range(num_iters):
        feature = get_feature(buffer, ae, DEVICE)  # (batch_size, dictionary_size)
        features[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, :] = feature

    firing_rate_per_feature = (features != 0).float().sum(dim=0) / total_inputs

    assert firing_rate_per_feature.shape[0] == dictionary_size

    mask = (firing_rate_per_feature > 0) & (firing_rate_per_feature < 0.5)
    idx = torch.nonzero(mask, as_tuple=False).squeeze()

    per_dim_stats = examine_dimension_chess(
        model,
        submodule,
        buffer,
        dictionary=ae,
        dims=idx[:MAX_DIMS],
        n_inputs=N_INPUTS,
        k=TOP_K + 1,
        batch_size=25,
        processing_device=torch.device("cpu"),
    )

    eval_results = evaluate(
        ae, buffer, max_len=context_length, batch_size=BATCH_SIZE, io="out", device=DEVICE
    )

    if save_results:
        pickle.dump(per_dim_stats, open(f"{autoencoder_path}per_dim_stats.pkl", "wb"))

    return per_dim_stats, eval_results


def compute_all_ae_stats(folder: str, save_results: bool = False):

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
        per_dim_stats, eval_results = get_ae_stats(folder, save_results=save_results)
        results = {}
        results["syntax"] = {}
        results["eval_results"] = eval_results

        for metric in syntax_metrics:
            metric_name = metric.__name__
            results["syntax"][metric_name] = chess_interp.syntax_analysis(
                per_dim_stats, TOP_K, TOP_K, MAX_DIMS, metric
            )
        results["board"] = chess_interp.board_analysis(
            per_dim_stats, TOP_K, TOP_K, MAX_DIMS, 0.99, board_metrics
        )
        total_results[folder] = results

        print(f"Finished {folder}")

        del per_dim_stats
        del eval_results
        gc.collect()

    total_results = chess_interp.serialize_results(total_results)
    total_results["hyperparameters"] = {}
    total_results["hyperparameters"]["n_inputs"] = N_INPUTS
    total_results["hyperparameters"]["top_k"] = TOP_K
    total_results["hyperparameters"]["max_dims"] = MAX_DIMS
    json.dump(total_results, open(f"total_results.json", "w"))


if __name__ == "__main__":
    compute_all_ae_stats("autoencoders/")
