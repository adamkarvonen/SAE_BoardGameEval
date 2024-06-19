from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE

model = LanguageModel(
    "EleutherAI/pythia-70m-deduped", device_map="cuda:0"  # this can be any Huggingface model
)

submodule = model.gpt_neox.layers[1].mlp  # layer 1 MLP
activation_dim = 512  # output dimension of the MLP
dictionary_size = 16 * activation_dim

# data much be an iterator that outputs strings
strs = [
    "This is some example data",
    "In real life, for training a dictionary",
    "you would need much more data than this",
] * 30000
data = iter(strs)
buffer = ActivationBuffer(
    data,
    model,
    submodule,
    submodule_output_dim=activation_dim,  # output dimension of the model component
    n_ctxs=3000,  # you can set this higher or lower dependong on your available memory
    device="cuda:0",  # doesn't have to be the same device that you train your autoencoder on
)  # buffer will return batches of tensors of dimension = submodule's output dimension

# train the sparse autoencoder (SAE)
ae = trainSAE(
    buffer,
    activation_dim,
    dictionary_size,
    lr=3e-4,
    sparsity_penalty=1e-3,
    device="cuda:0",
    resample_steps=25000,
    warmup_steps=1000,
)
