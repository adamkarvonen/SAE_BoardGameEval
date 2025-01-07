Our main SAE training script is `train_saes_parallel.py`. As an example of training Standard SAEs with p_annealing on layer 5 (0 indexed) of Chess-GPT, run:

`python circuits/sae_training/train_saes_parallel.py --game chess --layer 5 --trainer_type p_anneal --save_dir /workspace/SAE_BoardGameEval/autoencoders/chess_p_anneal_layer_5`

from the root directory.

By default, this command trains a sweep of 40 standard SAEs with p-annealing. There are other available trainers in the `dictionary_learning` repo, such as TopK and Gated, which can be selected using the `--trainer_type` flag. To adjust the size of the sweep, adjust the contents of `expansion_factor_` and `initial_sparsity_penalty_`. Note that these are set individually for each SAE trainer type.

Note that the sparsity penalties in the paper are set to use nonnormalized activations, which are currently used by default.

There is an optional flag `use_wandb` can be passed to `train_SAE()` if desired.