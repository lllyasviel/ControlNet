import cv2
import einops
import numpy as np
import torch
import random
# import gradio as gr
from pytorch_lightning import seed_everything
from ControlNet.annotator.util import resize_image, HWC3
from ControlNet.cldm.model import create_model, load_state_dict
from ControlNet.cldm.ddim_hacked import DDIMSampler
# import config as config
save_memory = False


class ControlSD:
    def __init__(self, model_config, checkpoint_path):
        self.model = create_model(model_config).cpu()
        self.model.load_state_dict(load_state_dict(checkpoint_path, location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

    def process(self, detected_map, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, 
                ddim_steps, guess_mode, strength, scale, seed, eta):
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
            detected_map = HWC3(detected_map)
            img = resize_image(detected_map, image_resolution)
            H, W, C = img.shape
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
            print("222")
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            print("333")
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)
            print("444")
            if save_memory:
                self.model.low_vram_shift(is_diffusing=False)
            print("555")
            cond = {
                "c_concat": [control], 
                "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
            }
            print("666")
            un_cond = {
                "c_concat": None if guess_mode else [control], 
                "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]
            }
            print("777")
            shape = (4, H // 8, W // 8)
            print("888")
            if save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

            samples, _ = self.ddim_sampler.sample(ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
                                                  unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)
            print("999")
            if save_memory:
                self.model.low_vram_shift(is_diffusing=False)
            print("1010")
            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            print("1111")
            results = [x_samples[i] for i in range(num_samples)]
        return [detected_map] + results


def create_control_sd():
    # model_config = '/home/data2/yangsp22/code/ControlNet/models/cldm_v21.yaml'
    # checkpoint_path = '/home/data2/yangsp22/code/finetune-output/sam-new/BDD-all/weights/lightning_logs/version_1/checkpoints/epoch=9-step=35139.ckpt'
    
    model_config = '/home/tmp/workspace/diffusion_app/backend/ControlNet/models/cldm_v21.yaml'
    checkpoint_path = '/home/tmp/workspace/diffusion_app/backend/epoch=9-step=35139.ckpt'
    
    # model = create_model('/home/tmp/workspace/diffusion_app/backend/ControlNet/models/cldm_v21.yaml').cpu()
    # model.load_state_dict(load_state_dict('/home/tmp/workspace/diffusion_app/backend/epoch=9-step=35139.ckpt', location='cuda'))
    return ControlSD(model_config, checkpoint_path)


control_sd_instance = create_control_sd()
process = control_sd_instance.process


# def create_gradio_interface(control_sd):
#     block = gr.Blocks().queue()
#     with block:
#         with gr.Row():
#             gr.Markdown("## Control Stable Diffusion with Segmentation Maps (Mask input)")
#         with gr.Row():
#             with gr.Column():
#                 input_image = gr.Image(source='upload', type="numpy")
#                 prompt = gr.Textbox(label="Prompt")
#                 run_button = gr.Button(label="Run")
#                 with gr.Accordion("Advanced options", open=False):
#                     num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
#                     image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
#                     strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
#                     guess_mode = gr.Checkbox(label='Guess Mode', value=False)
#                     detect_resolution = gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
#                     ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
#                     scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
#                     seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
#                     eta = gr.Number(label="eta (DDIM)", value=0.0)
#                     a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
#                     n_prompt = gr.Textbox(label="Negative Prompt",
#                                           value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
#             with gr.Column():
#                 result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')

#         ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
#         run_button.click(fn=control_sd.process, inputs=ips, outputs=[result_gallery])

#     return block


def main():
    control_sd = create_control_sd()
    block = create_gradio_interface(control_sd)
    block.launch(server_name='0.0.0.0')


if __name__ == "__main__":
    main()
