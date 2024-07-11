To train Othello SAEs on a single GPU, run the following command:

```
python othello_sae_trainer.py \
    --save_dir /share/u/can/SAE_BoardGameEval/autoencoders/transcoders/attn_out_sweep_all_layers_panneal_0703 \
    # --no_wandb_logging \
    # --dry_run \
    # --transcoder \
```

Note that save_dir requires a full path.
The first commented flag disables wandb logging. The second executes all code except training the SAE to look for potential errors. The third trains transcoders instead of SAEs trained to reconstruct activations.

By default, this command trains a standard SAE with p-annealing. There are other available trainers in the `dictionary_learning` repo, such as TopK and Gated.

To train Chess SAEs on a single GPU, run `python chess_sae_trainer.py` with the same flag options.

To train SAEs on multiple GPUs, TODO