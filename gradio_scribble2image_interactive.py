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
from ldm.models.diffusion.ddim import DDIMSampler


def get_device():
    if(torch.cuda.is_available()):
        return 'cuda'
    elif(torch.backends.mps.is_available()):
        return 'mps'
    else:
        return 'cpu'


device = get_device()
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location=device))
model = model.to(device)
ddim_sampler = DDIMSampler(model, device)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, scale, seed, eta):
    with torch.no_grad():
        img = resize_image(HWC3(input_image['mask'][:, :, 0]), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) > 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


def create_canvas(w, h):
    return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Interactive Scribbles")
    with gr.Row():
        with gr.Column():
            canvas_width = gr.Slider(label="Canvas Width", minimum=256, maximum=1024, value=512, step=1)
            canvas_height = gr.Slider(label="Canvas Height", minimum=256, maximum=1024, value=512, step=1)
            create_button = gr.Button(label="Start", value='Open drawing canvas!')
            input_image = gr.Image(source='upload', type='numpy', tool='sketch')
            gr.Markdown(value='Do not forget to change your brush width to make it thinner. (Gradio do not allow developers to set brush width so you need to do it manually.) '
                              'Just click on the small pencil icon in the upper right corner of the above block.')
            create_button.click(fn=create_canvas, inputs=[canvas_width, canvas_height], outputs=[input_image])
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=256)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
