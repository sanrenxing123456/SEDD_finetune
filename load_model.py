import os
import json
import torch
from model import SEDD
import utils
from model.ema import ExponentialMovingAverage
import graph_lib
import noise_lib

from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download


def _resolve_hf_file(model_id_or_path, filename):
    """Resolve a Hub file without relying on PyTorchModelHubMixin's format guess."""
    if os.path.isdir(model_id_or_path):
        path = os.path.join(model_id_or_path, filename)
        return path if os.path.isfile(path) else None
    try:
        return hf_hub_download(repo_id=model_id_or_path, filename=filename)
    except Exception:
        return None

def load_model_hf(dir, device):
    config_path = _resolve_hf_file(dir, "config.json")
    if config_path is None:
        raise FileNotFoundError(f"Could not find config.json for {dir}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = OmegaConf.create(json.load(handle))

    score_model = SEDD(config)
    safetensors_path = _resolve_hf_file(dir, "model.safetensors")
    pytorch_path = _resolve_hf_file(dir, "pytorch_model.bin")
    if safetensors_path is not None:
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path, device="cpu")
    elif pytorch_path is not None:
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"Could not find model.safetensors or pytorch_model.bin for {dir}"
        )
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    score_model.load_state_dict(state_dict)
    score_model = score_model.to(device)
    graph = graph_lib.get_graph(score_model.config, device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, graph, noise


def load_model_local(root_dir, device):
    cfg = utils.load_hydra_config_from_run(root_dir)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    ckpt_dir = os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
    loaded_state = torch.load(ckpt_dir, map_location=device)

    score_model.load_state_dict(loaded_state['model'])
    ema.load_state_dict(loaded_state['ema'])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model, graph, noise


def load_model(root_dir, device):
    hydra_config = os.path.join(root_dir, ".hydra", "config.yaml")
    if os.path.isfile(hydra_config):
        return load_model_local(root_dir, device)
    return load_model_hf(root_dir, device)
