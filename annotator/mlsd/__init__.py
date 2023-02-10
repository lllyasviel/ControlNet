import cv2
import numpy as np
import torch
import os

from einops import rearrange
from .models.mbv2_mlsd_tiny import  MobileV2_MLSD_Tiny
from .models.mbv2_mlsd_large import  MobileV2_MLSD_Large
from .utils import  pred_lines


model_path = './annotator/ckpts/mlsd_large_512_fp32.pth'
model = MobileV2_MLSD_Large()
model.load_state_dict(torch.load(model_path), strict=True)
model = model.cuda().eval()


def apply_mlsd(input_image, thr_v, thr_d):
    assert input_image.ndim == 3
    img = input_image
    img_output = np.zeros_like(img)
    try:
        with torch.no_grad():
            lines = pred_lines(img, model, [img.shape[0], img.shape[1]], thr_v, thr_d)
            for line in lines:
                x_start, y_start, x_end, y_end = [int(val) for val in line]
                cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
    except Exception as e:
        pass
    return img_output[:, :, 0]
