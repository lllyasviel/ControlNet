path_sd15 = './models/v1-5-pruned.ckpt'
path_sd15_with_control = './models/control_sd15_openpose.pth'
path_input = './models/anything-v3-full.safetensors'
path_output = './models/control_any3_openpose.ckpt'


import os


assert os.path.exists(path_sd15), 'Input path_sd15 does not exists!'
assert os.path.exists(path_sd15_with_control), 'Input path_sd15_with_control does not exists!'
assert os.path.exists(path_input), 'Input path_input does not exists!'
assert not os.path.exists(path_output), 'Output path_output already exists!'

from cldm.model import load_state_dict

sd15_state_dict = load_state_dict(path_sd15)
sd15_with_control_state_dict = load_state_dict(path_sd15_with_control)
input_state_dict = load_state_dict(path_input)

a = 0
