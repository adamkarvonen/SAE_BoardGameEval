from nnsight import NNsight
from transformer_lens import HookedTransformer
import itertools
from datasets import load_dataset

from circuits.othello_buffer import ActivationBuffer
import dictionary_learning.training as training


DEVICE = "cuda"

tf_model = HookedTransformer.from_pretrained("Baidicoot/Othello-GPT-Transformer-Lens")
model = NNsight(tf_model)
model = model.to(DEVICE)

submodule = model.blocks[5].hook_resid_post
activation_dim = 512  # output dimension of the layer
resample_steps = 50000


def othello_hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            yield x["tokens"][:59]  # Model seq_len is 60 - 1

    return gen()


data = othello_hf_dataset_to_generator("taufeeque/othellogpt")
buffer = ActivationBuffer(
    data,
    model,
    submodule,
    n_ctxs=8e3,
    ctx_len=59,
    refresh_batch_size=128,
    io="out",
    d_submodule=activation_dim,
    device=DEVICE,
)

learning_rates = [1e-3]
l1_penalties = [1e-4, 1e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1]
dictionary_sizes = [4 * activation_dim]

param_combinations = list(itertools.product(learning_rates, l1_penalties, dictionary_sizes))

trainer_configs = [
    {
        "trainer": training.StandardTrainer,
        "lr": lr,
        "l1_penalty": l1,
        "warmup_steps": 2000,
        "resample_steps": resample_steps,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
    }
    for lr, l1, dictionary_size in param_combinations
]

training.trainSAE(
    buffer,
    trainer_configs=trainer_configs,
    steps=300000,
    save_steps=100000,
    save_dir="",
    log_steps=1000,
)
