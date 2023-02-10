from annotator.uniformer.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from annotator.uniformer.mmseg.core.evaluation import get_palette


checkpoint_file = "annotator/ckpts/upernet_global_small.pth"
config_file = 'annotator/uniformer/exp/upernet_global_small/config.py'
model = init_segmentor(config_file, checkpoint_file).cuda()


def apply_uniformer(img):
    result = inference_segmentor(model, img)
    res_img = show_result_pyplot(model, img, result, get_palette('ade'), opacity=1)
    return res_img
