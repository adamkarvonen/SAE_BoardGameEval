import pickle
import torch

import circuits.eval_sae_as_classifier as eval_sae
import circuits.utils as utils

from circuits.dictionary_learning.evaluation import evaluate


def get_output_location(autoencoder_path: str, n_inputs: int) -> str:
    return autoencoder_path + f"n_inputs_{n_inputs}_evals.pkl"


def get_evals(
    autoencoder_path: str,
    n_inputs: int,
    batch_size: int,
    device: torch.device,
    data: dict,
    othello: bool = False,
    save_results: bool = True,
) -> dict:

    torch.set_grad_enabled(False)

    data, ae_bundle, pgn_strings, encoded_inputs = eval_sae.prep_data_ae_buffer_and_model(
        autoencoder_path,
        batch_size,
        data,
        device,
        n_inputs,
    )

    # Hacky way to check if the autoencoder comes from the GPT model Baidicoot/Othello-GPT-Transformer-Lens
    # If this fails, it will crash and not fail silently
    if othello and "random" not in autoencoder_path:
        eval_results = evaluate(
            ae_bundle.ae,
            ae_bundle.buffer,
            max_len=ae_bundle.context_length,
            batch_size=batch_size,
            io="out",
            device=device,
            tracer_args={},
        )
    else:
        eval_results = evaluate(
            ae_bundle.ae,
            ae_bundle.buffer,
            max_len=ae_bundle.context_length,
            batch_size=batch_size,
            io="out",
            device=device,
        )

    results = {}
    hyperparameters = {
        "n_inputs": n_inputs,
        "context_length": ae_bundle.context_length,
    }
    results["hyperparameters"] = hyperparameters
    results["eval_results"] = eval_results
    output_location = get_output_location(autoencoder_path, n_inputs)

    if save_results:
        results = utils.to_device(results, "cpu")
        with open(output_location, "wb") as f:
            pickle.dump(results, f)

    return results


def get_sae_group_evals(
    autoencoder_group_paths: list[str],
    device: str = "cuda",
    eval_inputs: int = 1000,
    batch_size: int = 10,
):

    # IMPORTANT NOTE: This is hacky (checks config 'ctx_len'), and means all autoencoders in the group must be for othello XOR chess
    othello = eval_sae.check_if_autoencoder_is_othello(autoencoder_group_paths[0])

    print("Constructing evaluation dataset...")

    data = eval_sae.construct_dataset(
        othello, [], split="train", n_inputs=(eval_inputs * 2), device=device
    )

    print("Starting evaluation...")

    for autoencoder_group_path in autoencoder_group_paths:
        print(f"Autoencoder group path: {autoencoder_group_path}")

        folders = utils.get_nested_folders(autoencoder_group_path)
        for autoencoder_path in folders:
            print("Evaluating autoencoder:", autoencoder_path)
            get_evals(
                autoencoder_path,
                eval_inputs,
                batch_size,
                device,
                data.copy(),
                othello=othello,
            )


if __name__ == "__main__":
    autoencoder_group_paths = ["autoencoders/othello_layer5_ef4/", "autoencoders/othello_layer0/"]
    autoencoder_group_paths = ["autoencoders/chess_layer5_large_sweep/"]
    autoencoder_group_paths = ["autoencoders/othello_layer5_ef4/"]

    get_sae_group_evals(autoencoder_group_paths)
