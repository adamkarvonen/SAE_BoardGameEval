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
from circuits.utils import get_feature, get_ae_bundle, AutoEncoderBundle


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

    data = hf_dataset_to_generator("adamkarvonen/chess_sae_test", streaming=False)
    ae_bundle = get_ae_bundle(autoencoder_path, device, data, batch_size, model_path)

    total_inputs = 8000
    assert total_inputs % batch_size == 0
    num_iters = total_inputs // batch_size

    # TODO This should be refactored so features is just shape (dictionary_size,) to reduce memory usage
    features = torch.zeros((total_inputs, ae_bundle.dictionary_size), device=device)
    for i in range(num_iters):
        feature = get_feature(
            ae_bundle.buffer, ae_bundle.ae, device
        )  # (batch_size, dictionary_size)
        features[i * batch_size : (i + 1) * batch_size, :] = feature

    firing_rate_per_feature = (features != 0).float().sum(dim=0) / total_inputs

    assert firing_rate_per_feature.shape[0] == ae_bundle.dictionary_size

    mask = (firing_rate_per_feature > 0) & (firing_rate_per_feature < 0.5)
    idx = torch.nonzero(mask, as_tuple=False).squeeze()

    per_dim_stats = examine_dimension_chess(
        ae_bundle.model,
        ae_bundle.submodule,
        ae_bundle.buffer,
        dictionary=ae_bundle.ae,
        max_length=ae_bundle.context_length,
        n_inputs=n_inputs,
        dims=idx[:max_dims],
        k=top_k + 1,
        batch_size=25,
        processing_device=torch.device("cpu"),
    )

    eval_results = evaluate(
        ae_bundle.ae,
        ae_bundle.buffer,
        max_len=ae_bundle.context_length,
        batch_size=batch_size,
        io="out",
        device=device,
    )

    if save_results:
        pickle.dump(per_dim_stats, open(f"{autoencoder_path}per_dim_stats.pkl", "wb"))

    return per_dim_stats, eval_results


def compute_all_ae_stats(
    folder: str,
    n_inputs: int = 5000,
    top_k: int = 30,
    max_dims: int = 4000,
    batch_size: int = 25,
    save_results: bool = False,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

        feature_dict = chess_interp.initialize_feature_dictionary(per_dim_stats)

        for metric in syntax_metrics:
            metric_name = metric.__name__
            results["syntax"][metric_name], feature_dict = chess_interp.syntax_analysis(
                per_dim_stats, top_k, top_k, max_dims, metric, feature_dict
            )
        results["board"], feature_dict = chess_interp.board_analysis(
            per_dim_stats, top_k, top_k, max_dims, 0.99, board_metrics, feature_dict
        )
        total_results[folder] = results

        with open(f"{folder}feature_dict.pkl", "wb") as f:
            pickle.dump(feature_dict, f)

        print(f"Finished {folder}")

        del per_dim_stats
        del eval_results
        del feature_dict
        gc.collect()

    total_results = chess_interp.serialize_results(total_results)
    total_results["hyperparameters"] = {}
    total_results["hyperparameters"]["n_inputs"] = n_inputs
    total_results["hyperparameters"]["top_k"] = top_k
    total_results["hyperparameters"]["max_dims"] = max_dims
    json.dump(total_results, open(f"total_results.json", "w"))


if __name__ == "__main__":
    compute_all_ae_stats("autoencoders/")
