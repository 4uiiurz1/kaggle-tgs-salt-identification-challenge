import numpy as np
import cv2

from skimage.io import imread

import torch
import torch.utils.data
from torchvision import datasets, models, transforms

from aug import *
from utils import depth_encode, coord_conv, pad


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=True, zs=None):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = imread(img_path)
        mask = imread(mask_path)

        image = image.astype('float32') / 255
        if 'train' in img_path:
            mask = (mask>65535/2).astype('float32')
        else:
            if self.args.loss == 'LovaszHingeLoss':
                mask = (mask>127).astype('float32')
            else:
                mask = mask.astype('float32') / 255

        if not self.args.pad:
            image = cv2.resize(image, (self.args.img_size, self.args.img_size))
            mask = cv2.resize(mask, (self.args.img_size, self.args.img_size))
        else:
            image = pad(image, self.args.img_size)
            mask = pad(mask, self.args.img_size)

        if self.aug:
            if self.args.aug == 1:
                image, mask = random_fliplr(image, mask)

            elif self.args.aug == 2:
                image, mask = random_fliplr(image, mask)
                image = random_erase(image)

            elif self.args.aug == 3:
                image, mask = random_fliplr(image, mask)
                image, mask = random_shift(image, mask, wrg=0.1, hrg=0.1, fill_mode='nearest')
                image = random_erase(image)

        if 'Res' in self.args.arch:
            means = [0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225]
            for i in range(3):
                image[:,:,i] = (image[:,:,i] - means[i]) / stds[i]

        if 'Inception' in self.args.arch:
            means = [0.5, 0.5, 0.5]
            stds = [0.5, 0.5, 0.5]
            for i in range(3):
                image[:,:,i] = (image[:,:,i] - means[i]) / stds[i]

        if self.args.depth:
            image = depth_encode(image)

        if self.args.coord_conv:
            image = coord_conv(image)

        image = image.transpose((2, 0, 1))
        mask = mask[:,:,np.newaxis]
        mask = mask.transpose((2, 0, 1))

        return image, mask
