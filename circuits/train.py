from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from circuits.nanogpt_to_hf_transformers import NanogptTokenizer, convert_nanogpt_model

import torch

device = torch.device("cuda:0")

tokenizer = NanogptTokenizer()
model = convert_nanogpt_model("lichess_6layers_ckpt_no_optimizer.pt", torch.device(device))
model = LanguageModel(model, device_map=device, tokenizer=tokenizer)

submodule = model.transformer.h[-1].mlp  # layer 1 MLP
activation_dim = 128  # output dimension of the MLP
dictionary_size = 16 * activation_dim

strs = [";1.e4 c5 2.Nf3 d6 3"] * 30000

data = iter(strs)
buffer = ActivationBuffer(
    data,
    model,
    submodule,
    submodule_output_dim=activation_dim,  # output dimension of the model component
    n_ctxs=3000,  # you can set this higher or lower dependong on your available memory
    device=device,  # doesn't have to be the same device that you train your autoencoder on
)  # buffer will return batches of tensors of dimension = submodule's output dimension

# train the sparse autoencoder (SAE)
ae = trainSAE(
    buffer,
    activation_dim,
    dictionary_size,
    lr=3e-4,
    sparsity_penalty=1e-3,
    device=device,
    resample_steps=25000,
    warmup_steps=1000,
)
