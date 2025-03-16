from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

#输入图片为分割图！！！

model = create_model('/home/tmp/workspace/old_diff/models/cldm_v21.yaml').cpu()
# /home/data2/yangsp22/code/finetune-output/sam-new/CODA-all/weights/lightning_logs/version_1/checkpoints/epoch=40-step=19999.ckpt
model.load_state_dict(load_state_dict('/home/tmp/workspace/diffusion_app/backend/epoch=9-step=35139.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(detected_map, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    print(detected_map.shape, "detected_map.shape")
    print(prompt, "prompt")
    print(a_prompt, "a_prompt")
    print(n_prompt, "n_prompt")
    print(num_samples, type(num_samples), "num_samples")
    print(image_resolution, type(image_resolution), "image_resolution")
    print(detect_resolution, type(detect_resolution), "detect_resolution")
    print(ddim_steps, type(ddim_steps), "ddim_steps")
    print(guess_mode, type(guess_mode), "guess_mode")
    print(strength, type(strength), "strength")
    print(scale, type(scale), "scale")
    print(seed, type(seed), "seed")
    print(eta, type(eta), "eta")
    print("111")
    with torch.no_grad():
        detected_map = HWC3(detected_map) # 以输入图片1020 x 1920为例，(h, w, c) （高，宽，channel）(1020, 1920, 3)
        img = resize_image(detected_map, image_resolution) # (1020, 1920, 3) -> (512, 960, 3)
        H, W, C = img.shape # 512, 960, 3
        print(11111)
        # cv2.INTER_NEAREST：最邻近插值
        # 使用 最邻近插值法 将图像 detected_map 调整为指定大小 (W, H)
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0 # shape=torch.size([512,960,3]), dtype=torch.float32, 值在0-1区间
        control = torch.stack([control for _ in range(num_samples)], dim=0) # num_samples=1, shape=torch.size([1,512,960,3])
        control = einops.rearrange(control, 'b h w c -> b c h w').clone() # b h w c -> b c h w

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed) #随机种子

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        
        # !!!
        # obstruction_prompt="A construction zone occupies part of the right lane, with cones, workers, and safety signs."
        
        # !!!
        # cond（正向控制条件）：图片转成control，prompt+a_prompt作为交叉注意力的控制条件
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        # cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], 
        #         "obstruction_c_crossattn": [model.get_learned_conditioning([obstruction_prompt] * num_samples)]}
        
        # !!!
        # un_cond（反向控制条件）：图片转成control，n_prompt作为交叉注意力的负面控制条件
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        # un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], 
        #         "obstruction_c_crossattn": None}
        
        shape = (4, H // 8, W // 8) # H 512, W 960 -> shape (4, 64, 120)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # 控制权重
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        # samples -> shape=torch.size([1,4,64,120])
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples) # shape=torch.size([1,3,512,960])
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)] # list
    return [detected_map] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Segmentation Maps (Mask input)")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                detect_resolution = gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
