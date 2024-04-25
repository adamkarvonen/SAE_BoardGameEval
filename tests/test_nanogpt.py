from context import circuits

from circuits import nanogpt_to_hf_transformers

from transformers import GPT2LMHeadModel, GPT2Config

import pickle
import torch

META_PATH = "models/meta.pkl"
MODEL_PATH = "models/lichess_6layers_ckpt_no_optimizer.pt"


def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def test_tokenizer():
    test_str = ";1.e4 e5 2.Nf3"

    meta = load_meta()
    stoi, itos = meta["stoi"], meta["itos"]

    tokenizer = nanogpt_to_hf_transformers.NanogptTokenizer()

    enc = tokenizer.encode(test_str)
    enc = torch.tensor(enc).unsqueeze(0)
    nanogpt_enc = torch.tensor([stoi[c] for c in test_str]).unsqueeze(0)

    # tokenizer.save_vocabulary("models")

    assert torch.equal(enc, nanogpt_enc)


def test_model_conversion():
    model_name = "models/lichess_8layers_ckpt_no_optimizer.pt"
    device = torch.device("cpu")

    nanogpt_model, nanogpt_config = nanogpt_to_hf_transformers.get_nanogpt_model_and_config(
        model_name, device
    )

    hf_model = nanogpt_to_hf_transformers.nanogpt_to_hf(nanogpt_model, nanogpt_config)

    test_str = ";1.e4 e5 2.Nf3"
    meta = load_meta()
    stoi, itos = meta["stoi"], meta["itos"]
    enc = torch.tensor([stoi[c] for c in test_str]).unsqueeze(0)

    hf_model.save_pretrained("models")

    hf_response = hf_model.generate(enc, max_length=len(test_str) + 40, do_sample=False)
    nanogpt_response = nanogpt_model.generate(enc, max_new_tokens=40, temperature=0.0)

    assert torch.equal(hf_response[0], nanogpt_response[0])


def test_saved_model():

    model_name = "models/lichess_8layers_ckpt_no_optimizer.pt"
    device = torch.device("cpu")

    nanogpt_model, nanogpt_config = nanogpt_to_hf_transformers.get_nanogpt_model_and_config(
        model_name, device
    )

    model = GPT2LMHeadModel.from_pretrained("models")
    test_str = ";1.e4 e5 2.Nf3"
    meta = load_meta()
    stoi, itos = meta["stoi"], meta["itos"]
    enc = torch.tensor([stoi[c] for c in test_str]).unsqueeze(0)
    hf_response = model.generate(enc, max_length=len(test_str) + 40, do_sample=False)
    nanogpt_response = nanogpt_model.generate(enc, max_new_tokens=40, temperature=0.0)

    assert torch.equal(hf_response[0], nanogpt_response[0])
