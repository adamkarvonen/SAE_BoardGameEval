Our main SAE training script is `train_saes_parallel.py`. As an example of training Standard SAEs with p_annealing on layer 5 (0 indexed) of Chess-GPT, run:

`python circuits/sae_training/train_saes_parallel.py --game chess --layer 5 --trainer_type p_anneal --save_dir /workspace/SAE_BoardGameEval/autoencoders/chess_p_anneal_layer_5`

from the root directory.

By default, this command trains a sweep of 40 standard SAEs with p-annealing. There are other available trainers in the `dictionary_learning` repo, such as TopK and Gated, which can be selected using the `--trainer_type` flag.