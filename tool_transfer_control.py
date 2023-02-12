path_sd15 = './models/v1-5-pruned.ckpt'
path_sd15_with_control = './models/control_sd15_openpose.pth'
path_input = './models/anything-v3-full.safetensors'
path_output = './models/control_any3_openpose.ckpt'


import os


assert os.path.exists(path_sd15), 'Input path_sd15 does not exists!'
assert os.path.exists(path_sd15_with_control), 'Input path_sd15_with_control does not exists!'
assert os.path.exists(path_input), 'Input path_input does not exists!'
assert os.path.exists(os.path.dirname(path_output)), 'Output folder not exists!'


import torch
from share import *
from cldm.model import load_state_dict


sd15_state_dict = load_state_dict(path_sd15)
sd15_with_control_state_dict = load_state_dict(path_sd15_with_control)
input_state_dict = load_state_dict(path_input)


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


keys = sd15_with_control_state_dict.keys()

final_state_dict = {}
for key in keys:
    is_control, node_name = get_node_name(key, 'control_')
    if is_control:
        pass
    else:
        pass

torch.save(final_state_dict, path_output)
print('Transferred model saved at ' + path_output)
