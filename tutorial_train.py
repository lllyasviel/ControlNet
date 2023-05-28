from share import *

from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from s2s_dataset import S2sDataSet
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torchvision import transforms


# Configs
resume_path = './models/v1-5-pruned.ckpt'
input_directory = Path('/Users/jakubgalik/repos/ControlNet/dataset/train')
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
size = 512, 512


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
# model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = S2sDataSet(input_directory, size)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(precision=32, callbacks=[logger]) #, accelerator="mps", devices=1)


# Train!
trainer.fit(model, dataloader)
