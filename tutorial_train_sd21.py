from share import *
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from s2s_dataset import S2sDataSet
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Script for data conversion.')
    parser.add_argument('--input-dir', type=Path, required=True, help='Path to input data directory.')
    parser.add_argument('--resume-path', type=Path, help='Path to checkpoint file.')
    parser.add_argument('--batch-size', type=int, default=4, help='Size of batch.')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate value.')
    parser.add_argument('--logger-freq', type=int, default=300, help='Logging frequency in steps.')
    parser.add_argument('--sd-locked', type=bool, default=True, help='If stable diffusion layers locked.')
    parser.add_argument('--only-mid-control', type=bool, default=False, help='Only medium control.')
    parser.add_argument('--only-mid-control', nargs=2, type=int, default=[512, 512], help='Only medium control.')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers.')
    return parser.parse_args()


def main():
    args = parse_args()
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(args.resume_path, location='cpu'))
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    # Misc
    dataset = S2sDataSet(args.input_dir, args.size)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=args.logger_freq)
    trainer = pl.Trainer(accelerator='gpu', precision=32, callbacks=[logger])

    # Train!
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
