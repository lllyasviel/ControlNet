import sys
import os

import torch
from share import *
from cldm.model import create_model
import hydra
from pathlib import Path


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


@hydra.main(config_path=str(Path.cwd()), config_name='config', version_base=None)
def main(cfg):


    assert os.path.exists(cfg.model_preparation.sd_init_model), 'Input model does not exist.'
    assert not os.path.exists(cfg.model_preparation.init_controlnet_model), 'Output filename already exists.'
    assert os.path.exists(os.path.dirname(cfg.model_preparation.init_controlnet_model)), 'Output path is not valid.'
    model = create_model(config_path=cfg.model)

    pretrained_weights = torch.load(cfg.model_preparation.sd_init_model)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    scratch_dict = model.state_dict()

    target_dict = {}
    for k in scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = 'model.diffusion_' + name
        else:
            copy_k = k
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')

    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), cfg.model_preparation.init_controlnet_model)
    print('Done.')


if __name__ == '__main__':
    main()
