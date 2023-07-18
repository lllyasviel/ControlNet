from share import *
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from s2s_dataset import S2sDataSet
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path='../', config_name='config')
def main(cfg: DictConfig):
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(cfg.model).cpu()
    model.load_state_dict(load_state_dict(cfg.resume_path, location='cpu'))
    model.learning_rate = cfg.learning_rate
    model.sd_locked = cfg.sd_locked
    model.only_mid_control = cfg.only_mid_control

    # Misc
    dataset = S2sDataSet(cfg.input_dir, cfg.size)
    dataloader = DataLoader(dataset, num_workers=cfg.workers, batch_size=cfg.batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=cfg.logger_freq)
    trainer = pl.Trainer(accelerator='gpu', precision=32, callbacks=[logger])

    # Train!
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
