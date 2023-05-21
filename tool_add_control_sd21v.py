import sys
import os

assert len(sys.argv) == 3, "Args are wrong."

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), "Input model does not exist."
assert not os.path.exists(output_path), "Output filename already exists."
assert os.path.exists(os.path.dirname(output_path)), "Output path is not valid."

import torch
from share import *
from cldm.model import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ""
    p = name[: len(parent_name)]
    if p != parent_name:
        return False, ""
    return True, name[len(parent_name) :]


model = create_model(config_path="./models/cldm_v21v.yaml")

pretrained_weights = torch.load(input_path)
if "state_dict" in pretrained_weights:
    pretrained_weights = pretrained_weights["state_dict"]

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, "control_")
    if is_control:
        copy_k = "model.diffusion_" + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f"These weights are newly added: {k}")

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print("Done.")
