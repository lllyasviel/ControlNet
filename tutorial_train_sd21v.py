from share import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = "./models/control_sd21v_ini.ckpt"
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model("./models/cldm_v21v.yaml").cpu()
model.load_state_dict(load_state_dict(resume_path, location="cpu"))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

image_logger = ImageLogger(".", batch_frequency=logger_freq)
wandb_logger = WandbLogger()
tb_logger = TensorBoardLogger("tb_logs")

checkpoint_callback = ModelCheckpoint(
    every_n_train_epochs=10,
    monitor=None,
    dirpath="./output/circle-1",
    filename="circle-{epoch}",
    save_top_k=5,  # save top 5 models
    save_last=True,
    verbose=True,
)

trainer = pl.Trainer(
    gpus=1,
    precision=32,
    logger=[wandb_logger, tb_logger],
    callbacks=[checkpoint_callback, image_logger],
)


# Train!
trainer.fit(model, dataloader)
