import os
import torch

from omegaconf import OmegaConf
from control_net.ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)

    # Remove position_ids from CLIP text model as it's now a buffer, not a parameter
    position_id_key = "cond_stage_model.transformer.text_model.embeddings.position_ids"
    if position_id_key in state_dict:
        print(f"Removing {position_id_key} from state_dict (now handled as buffer)")
        del state_dict[position_id_key]

    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
