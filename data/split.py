import yaml
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Split dataset')
    parser.add_argument('--split-file', type=Path, required=True)
    parser.add_argument('--scheme-dir', type=Path, required=True)
    parser.add_argument('--simulation-dir', type=Path, help='If given, then path in splityaml is overwritten.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.split_file) as file:
        split_data = yaml.safe_load(file)

    trainsplit = split_data[args.split]
    scheme_outdir = args.output_dir / args.split / 'scheme'
    simulation_outdir = args.output_dir / args.split / 'simulation'
    scheme_outdir.mkdir(parents=True, exist_ok=True)
    simulation_outdir.mkdir(parents=True, exist_ok=True)
    for simulation_path in tqdm(trainsplit):
        simulation_path = Path(simulation_path)
        if args.simulation_dir:
            simulation_path = args.simulation_dir / simulation_path.name
        scheme_path = args.scheme_dir / simulation_path.name
        shutil.copy(simulation_path, args.output_dir / args.split / 'scheme' / simulation_path.name)
        shutil.copy(scheme_path, args.output_dir / args.split / 'simulation' / simulation_path.name)


if __name__ == '__main__':
    main()
