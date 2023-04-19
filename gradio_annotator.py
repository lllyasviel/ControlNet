import gradio as gr
import torch

from annotator.util import resize_image, HWC3


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


model_canny = None
device = get_device()


def canny(img, res, l, h):
    img = resize_image(HWC3(img), res)
    global model_canny
    if model_canny is None:
        from annotator.canny import CannyDetector
        model_canny = CannyDetector()
    result = model_canny(img, l, h)
    return [result]


model_hed = None


def hed(img, res):
    img = resize_image(HWC3(img), res)
    global model_hed
    if model_hed is None:
        from annotator.hed import HEDdetector
        model_hed = HEDdetector(device)
    result = model_hed(img)
    return [result]


model_mlsd = None


def mlsd(img, res, thr_v, thr_d):
    img = resize_image(HWC3(img), res)
    global model_mlsd
    if model_mlsd is None:
        from annotator.mlsd import MLSDdetector
        model_mlsd = MLSDdetector(device)
    result = model_mlsd(img, thr_v, thr_d)
    return [result]


model_midas = None


def midas(img, res, a):
    img = resize_image(HWC3(img), res)
    global model_midas
    if model_midas is None:
        from annotator.midas import MidasDetector
        model_midas = MidasDetector(device)
    results = model_midas(img, a)
    return results


model_openpose = None


def openpose(img, res, has_hand):
    img = resize_image(HWC3(img), res)
    global model_openpose
    if model_openpose is None:
        from annotator.openpose import OpenposeDetector
        model_openpose = OpenposeDetector()
    result, _ = model_openpose(img, has_hand)
    return [result]


model_uniformer = None


def uniformer(img, res):
    img = resize_image(HWC3(img), res)
    global model_uniformer
    if model_uniformer is None:
        from annotator.uniformer import UniformerDetector
        model_uniformer = UniformerDetector(device)
    result = model_uniformer(img)
    return [result]


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Canny Edge")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            low_threshold = gr.Slider(label="low_threshold", minimum=1, maximum=255, value=100, step=1)
            high_threshold = gr.Slider(label="high_threshold", minimum=1, maximum=255, value=200, step=1)
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=canny, inputs=[input_image, resolution, low_threshold, high_threshold], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## HED Edge")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=hed, inputs=[input_image, resolution], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## MLSD Edge")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            value_threshold = gr.Slider(label="value_threshold", minimum=0.01, maximum=2.0, value=0.1, step=0.01)
            distance_threshold = gr.Slider(label="distance_threshold", minimum=0.01, maximum=20.0, value=0.1, step=0.01)
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=384, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=mlsd, inputs=[input_image, resolution, value_threshold, distance_threshold], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## MIDAS Depth and Normal")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            alpha = gr.Slider(label="alpha", minimum=0.1, maximum=20.0, value=6.2, step=0.01)
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=384, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=midas, inputs=[input_image, resolution, alpha], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## Openpose")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            hand = gr.Checkbox(label='detect hand', value=False)
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=openpose, inputs=[input_image, resolution, hand], outputs=[gallery])


    with gr.Row():
        gr.Markdown("## Uniformer Segmentation")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=uniformer, inputs=[input_image, resolution], outputs=[gallery])


block.launch(server_name='0.0.0.0')
