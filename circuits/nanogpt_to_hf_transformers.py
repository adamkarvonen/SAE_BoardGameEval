import torch
from transformers import GPT2Config, GPT2LMHeadModel
import os
from transformers.tokenization_utils import PreTrainedTokenizer
import pickle

from .nanogpt import GPT, GPTConfig


def nanogpt_to_hf(nanogpt_model, nanogpt_config: GPTConfig):

    config = GPT2Config(
        vocab_size=nanogpt_config.vocab_size,  # standard for GPT-2 models
        n_positions=nanogpt_config.block_size,  # standard max sequence length for GPT-2
        n_ctx=nanogpt_config.block_size,  # context window size, usually equals n_positions
        n_embd=nanogpt_config.n_embd,  # model embedding dimensionality
        n_layer=nanogpt_config.n_layer,  # number of layers
        n_head=nanogpt_config.n_head,  # number of attention heads per layer
        resid_pdrop=nanogpt_config.dropout,
        embd_pdrop=nanogpt_config.dropout,
        attn_pdrop=nanogpt_config.dropout,
    )

    # Initialize a new Hugging Face model with the specified configuration
    model_hf = GPT2LMHeadModel(config)

    sd_nanogpt = nanogpt_model
    sd_hf = model_hf.state_dict()

    # Adjust the names and orientation of weights as necessary
    for key in sd_hf.keys():
        if (
            "attn.c_attn.weight" in key
            or "attn.c_proj.weight" in key
            or "mlp.c_fc.weight" in key
            or "mlp.c_proj.weight" in key
        ):
            # Transpose the weights if they are from a Linear layer in nanogpt assumed to be equivalent to Conv1D in HF
            sd_hf[key] = sd_nanogpt[key].t()
        else:
            # Direct copy for other parameters
            sd_hf[key] = sd_nanogpt[key]

    # Load the modified state dict back into the Hugging Face model
    model_hf.load_state_dict(sd_hf)

    return model_hf


def add_zeros_bias_to_state_dict(
    state_dict,
    device,
    config: GPTConfig,
):
    """If the nanogpt model does not have bias, add zeros bias to the state_dict for compatibility
    with the Hugging Face GPT2LMHeadModel."""

    if config.bias:
        # If the model already has bias, return the state_dict as is
        return state_dict, config

    config.bias = True

    state_dict["transformer.ln_f.bias"] = torch.zeros_like(state_dict["transformer.ln_f.weight"])

    for i in range(config.n_layer):
        layer_key = f"transformer.h.{i}"

        state_dict[f"{layer_key}.ln_1.bias"] = torch.zeros_like(
            state_dict[f"{layer_key}.ln_1.weight"]
        )
        state_dict[f"{layer_key}.ln_2.bias"] = torch.zeros_like(
            state_dict[f"{layer_key}.ln_2.weight"]
        )

        mlp_bias_shape = state_dict[f"{layer_key}.mlp.c_fc.weight"].shape[0]

        assert mlp_bias_shape == config.n_embd * 4

        state_dict[f"{layer_key}.mlp.c_fc.bias"] = torch.zeros(mlp_bias_shape, device=device)
        state_dict[f"{layer_key}.mlp.c_proj.bias"] = torch.zeros(config.n_embd, device=device)

        state_dict[f"{layer_key}.attn.c_attn.bias"] = torch.zeros(config.n_embd * 3, device=device)
        state_dict[f"{layer_key}.attn.c_proj.bias"] = torch.zeros(config.n_embd, device=device)

    return state_dict, config


def get_nanogpt_model_and_config(model_name: str, device: torch.device) -> tuple[GPT, GPTConfig]:
    model_path = os.path.join("models", model_name)
    checkpoint = torch.load(model_path, map_location=device)
    nanogpt_config = GPTConfig(**checkpoint["model_args"])

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    state_dict, nanogpt_config = add_zeros_bias_to_state_dict(state_dict, device, nanogpt_config)
    nanogpt_model = GPT(nanogpt_config)
    nanogpt_model.load_state_dict(state_dict)

    return nanogpt_model, nanogpt_config


def convert_nanogpt_model(model_name: str, device: torch.device) -> GPT2LMHeadModel:
    nanogpt_model, nanogpt_config = get_nanogpt_model_and_config(model_name, device)
    model_hf = nanogpt_to_hf(nanogpt_model, nanogpt_config)
    return model_hf


class NanogptTokenizer(PreTrainedTokenizer):
    """
    Adapted from CanineTokenizer in transformers package.
    """

    def __init__(
        self,
        add_prefix_space=False,
        model_max_length=2048,
        **kwargs,
    ):
        meta_path = "models/meta.pkl"

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        stoi, itos = meta["stoi"], meta["itos"]
        self._vocab_size = len(stoi)
        self._num_special_tokens = 0
        self.stoi = stoi
        self.itos = itos

        super().__init__(
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self):
        """Not sure what this function is for but it throws an error if not implemented."""
        return self.stoi
        vocab = {chr(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize a string (i.e. perform character splitting)."""
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (i.e. a Unicode character) in an id (i.e. its integer Unicode code point value)."""
        try:
            return self.stoi[token]
        except:
            raise ValueError(f"invalid token: '{token}'")

    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts a Unicode code point (integer) in a token (str). In case it's a special code point, convert to
        human-readable format.
        """
        try:
            return self.itos[index]
        except:
            raise ValueError(f"invalid id: {index}")

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)
