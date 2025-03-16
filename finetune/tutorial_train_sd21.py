from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
import sys
sys.path.append("/home/data2/yangsp22/code/ControlNet")
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# 运行指令参考
# python tutorial_train_sd21.py
# CUDA_VISIBLE_DEVICES=3,4,5,6,7 nohup python /home/data2/yangsp22/code/ControlNet/finetune/tutorial_train_sd21.py > /home/data2/yangsp22/code/finetune-output/sam-new/CODA-all/output2.log 2>&1 &

# Configs
resume_path = '/home/data2/yangsp22/code/ControlNet/models/control_sd21_ini.ckpt'
batch_size = 1
accumulate_grad_batches = 4
logger_freq = 300
learning_rate = 1e-5
max_steps = 20000 #训练多少步后停止，-1表示没有限制，一步指的是一次学习batch_size个样本 # 9000steps
max_epochs = -1 #训练多少轮后停止，-1表示没有限制，一轮指的是每个样本都学习过一次了 # 8epochs
sd_locked = True
only_mid_control = False
weights_save_path = '/home/data2/yangsp22/code/finetune-output/sam-new/CODA-all/weights'
default_root_dir = '/home/data2/yangsp22/code/finetune-output/sam-new/CODA-all'


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('/home/data2/yangsp22/code/ControlNet/models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=5, precision=32, callbacks=[logger], accumulate_grad_batches=accumulate_grad_batches,
                     max_steps=max_steps, max_epochs=max_epochs, 
                     weights_save_path=weights_save_path, default_root_dir=default_root_dir, strategy="ddp")


# Train!
# if __name__ == '__main__': 
trainer.fit(model, dataloader)
