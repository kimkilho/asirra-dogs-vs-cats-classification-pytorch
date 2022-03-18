import os
import os.path as osp
import random

import numpy as np
from skimage import io, color, transform
import torch
from torch.utils.data import Dataset


class AsirraDataset(Dataset):
    """Asirra dataset class"""

    def __init__(self, img_dir, img_filenames, class_names, transform=None):
        self.img_dir = img_dir
        self.img_filenames = img_filenames
        self.class_label_dict = {class_name: l for l, class_name in enumerate(class_names)}
        self.transform = transform

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_filename = self.img_filenames[idx]
        img_path = osp.join(self.img_dir, img_filename)
        image = io.imread(img_path)
        label = self.class_label_dict[img_filename.split('.')[0]]

        if self.transform:
            image = self.transform(image)

        return image, label


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


class RandomFlip(object):
    """Randomly flip the image in a sample horizontally or vertically."""

    def __init__(self, mode='horizontal_and_vertical'):
        self.mode = mode

    def __call__(self, image):
        if self.mode == 'horizontal_and_vertical' or self.mode == 'vertical':
            pv = random.random()
            if pv < 0.5:
                image = vflip(image)
        if self.mode == 'horizontal_and_vertical' or self.mode == 'horizontal':
            ph = random.random()
            if ph < 0.5:
                image = hflip(image)

        return image


def _is_numpy_image(image):
    return isinstance(image, np.ndarray) and (image.ndim in {2, 3})


def hflip(image):
    if not _is_numpy_image(image):
        raise TypeError('image should be numpy.ndarray. Got {}.'.format(type(image)))
    return np.ascontiguousarray(np.fliplr(image))


def vflip(image):
    if not _is_numpy_image(image):
        raise TypeError('image should be numpy.ndarray. Got {}.'.format(type(image)))
    return np.ascontiguousarray(np.flipud(image))
