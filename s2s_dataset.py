from torch.utils.data import Dataset
import cv2
from pathlib import Path
import yaml


class S2sDataSet(Dataset):
    def __init__(self, split_file_path, mode="train"):
        with open(split_file_path, 'r', encoding='utf-8') as split_file:
            split_data = yaml.safe_load(split_file)
        if mode in ['train', 'val']:
            self.files = split_data[mode]
        else:
            ValueError(f'Wrong data loader mode.')

    def __getitem__(self, index):
        simulation_path = Path(self.files[index % len(self.files)])
        scheme_path = simulation_path.parent.parent / 'scheme' / simulation_path.name
        prompt = 'flat design'


        source = cv2.imread(str(scheme_path))
        target = cv2.imread(str(simulation_path))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        return dict(jpg=target, txt=prompt, hint=source)

    def __len__(self):
        return len(self.files)