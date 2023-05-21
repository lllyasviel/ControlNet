import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
import wandb


class ImageLogger(Callback):
    def __init__(
        self,
        save_dir: str,
        batch_frequency=2000,
        max_images=4,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(self.save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k, global_step, current_epoch, batch_idx
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_wandb(self, images):
        concatenated_images = {}

        keys = ["conditioning", "control", "reconstruction", "samples"]

        for k in keys:
            for image_key in images:
                if image_key.startswith(k):
                    grid = torchvision.utils.make_grid(images[image_key], nrow=4)
                    if self.rescale:
                        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.numpy()
                    grid = (grid * 255).astype(np.uint8)

                    concatenated_images[k] = grid

        image_array = [
            concatenated_images["conditioning"],
            concatenated_images["control"],
            concatenated_images["samples"],
            concatenated_images["reconstruction"],
        ]

        # to wandb
        concatenated = np.vstack(image_array)
        logged_image = wandb.Image(Image.fromarray(concatenated))

        return logged_image

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            self.log_local(
                split, images, pl_module.global_step, pl_module.current_epoch, batch_idx
            )

            wandb_logger = None
            for logger in pl_module.logger:
                if isinstance(logger, WandbLogger):
                    wandb_logger = logger
                    break

            if wandb_logger is not None:
                logged_image = self.log_wandb(images)
                # to wandb
                wandb_logger.experiment.log(
                    {
                        "sample_images": logged_image,
                    }
                )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
