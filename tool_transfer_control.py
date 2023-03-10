import argparse
import os
import sys

import torch
from share import *
from cldm.model import load_state_dict


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

def parse_args():
    parser = argparse.ArgumentParser(description='Transfer weights from one model to another')

    parser.add_argument('--path_sd15', required=True, type=str, help='Path to sd15 model')
    parser.add_argument('--path_sd15_with_control', required=True, type=str, help='Path to sd15 model with control')
    parser.add_argument('--path_input', required=True, type=str, help='Path to input model')
    parser.add_argument('--path_output', required=True, type=str, help='Path to output transferred model')

    return parser.parse_args()


def main():
    args = parse_args()

    for path in [args.path_sd15, args.path_sd15_with_control, args.path_input]:
        if not os.path.exists(path):
            print(f"Error: Input path '{path}' does not exist!")
            sys.exit(1)

    if not os.path.exists(os.path.dirname(args.path_output)):
        print("Error: Output folder does not exist!")
        sys.exit(1)

    sd15_state_dict = load_state_dict(args.path_sd15)
    sd15_with_control_state_dict = load_state_dict(args.path_sd15_with_control)
    input_state_dict = load_state_dict(args.path_input)

    keys = sd15_with_control_state_dict.keys()

    final_state_dict = {}
    for key in keys:
        is_first_stage, _ = get_node_name(key, 'first_stage_model')
        is_cond_stage, _ = get_node_name(key, 'cond_stage_model')
        if is_first_stage or is_cond_stage:
            final_state_dict[key] = input_state_dict[key]
            continue
        p = sd15_with_control_state_dict[key]
        is_control, node_name = get_node_name(key, 'control_')
        if is_control:
            sd15_key_name = 'model.diffusion_' + node_name
        else:
            sd15_key_name = key
        if sd15_key_name in input_state_dict:
            p_new = p + input_state_dict[sd15_key_name] - sd15_state_dict[sd15_key_name]
        else:
            p_new = p
        final_state_dict[key] = p_new

    torch.save(final_state_dict, args.path_output)
    print(f'Transferred model saved at {args.path_output}')


if __name__ == '__main__':
    main()
