import os
import os.path as osp
import argparse

from skimage import io
from skimage import transform
import torch
import torchvision.transforms as transforms

from azureml.core import (
    Workspace, Model
)


CHANNEL_MEAN = [0.485, 0.456, 0.406]
CHANNEL_STD = [0.229, 0.224, 0.225]
CLASS_NAMES_DICT = {
    'asirra': ['cat', 'dog'],
}


class Rescale(object):
    """Rescale the image in a sample to a given size."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared_input_data', type=str, help='prepared input data')
    parser.add_argument('--output_dir', type=str, help='prediction output directory')

    return parser.parse_args()


if __name__ == '__main__':
    dataset_name = 'asirra'
    args = parse_args()
    data_dir = args.prepared_input_data
    output_dir = args.output_dir

    print('Data received: {}'.format(data_dir))

    # Set class names and image transformation functions
    class_names = CLASS_NAMES_DICT[dataset_name]
    transform_fn = transforms.Compose([
        Rescale((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD)
    ])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    filename = os.listdir(data_dir)[0]
    img_file_path = osp.join(data_dir, filename)
    image = torch.unsqueeze(transform_fn(io.imread(img_file_path)), 0).float().to(device)
    # transformed image, shape: (1, 3, H, W)

    model_path = Model.get_model_path(model_name='asirra-ResNet50')
    model_ft = torch.load(model_path).to(device)

    with torch.no_grad():
        pred = model_ft(image)[0].cpu().numpy()

    output_file_path = osp.join(output_dir, 'prediction.txt')
    with open(output_file_path, 'w') as fid:
        fid.write(str(pred[0]) + ', ' + str(pred[1]))
