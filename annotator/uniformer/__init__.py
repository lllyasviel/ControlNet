from annotator.uniformer.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from annotator.uniformer.mmseg.core.evaluation import get_palette


class UniformerDetector:
    def __init__(self):
        checkpoint_file = "annotator/ckpts/upernet_global_small.pth"
        config_file = 'annotator/uniformer/exp/upernet_global_small/config.py'
        self.model = init_segmentor(config_file, checkpoint_file).cuda()

    def __call__(self, img):
        result = inference_segmentor(self.model, img)
        res_img = show_result_pyplot(self.model, img, result, get_palette('ade'), opacity=1)
        return res_img
