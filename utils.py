import random

import numpy as np
import cv2


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def rle_encode(mask_image):
    mask_image = np.rot90(mask_image[:, ::-1])
    pixels = mask_image.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]

    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    img = img.reshape(shape)
    img = np.rot90(img[:, ::-1], 1)

    return img


def depth_encode(img):
    h, w, _ = img.shape
    for i, d in enumerate(np.linspace(0, 1, h)):
        img[i, :, 1] = d
    img[:, :, 2] = img[:, :, 0] * img[:, :, 1]

    return img


def coord_conv(img):
    h, w, _ = img.shape
    for i, d in enumerate(np.linspace(0, 1, h)):
        img[i, :, 1] = d
    for i, d in enumerate(np.linspace(0, 1, w)):
        img[:, i, 2] = d

    return img


def resize_pad_encode(img, size1, size2, mode='edge'):
    img = cv2.resize(img, (size1, size1))
    pad = (size2 - size1) // 2
    if img.ndim == 2:
        img = np.pad(img, ((pad, size2 - size1 - pad), (pad, size2 - size1 - pad)), mode='edge')
    elif img.ndim == 3:
        img = np.pad(img, ((pad, size2 - size1 - pad), (pad, size2 - size1 - pad), (0, 0)), mode='edge')

    return img


def resize_pad_decode(img, size1, size2):
    pad = (size2 - size1) // 2
    img = img[pad:pad+size1, pad:pad+size1]
    img = cv2.resize(img, (101, 101))


def pad(img, size):
    pad_lu = (size - img.shape[0]) // 2
    pad_rd = size - img.shape[0] - pad_lu
    if img.ndim == 2:
        img = np.pad(img, ((pad_lu, pad_rd), (pad_lu, pad_rd)), mode='edge')
    elif img.ndim == 3:
        img = np.pad(img, ((pad_lu, pad_rd), (pad_lu, pad_rd), (0, 0)), mode='edge')

    return img


def crop(img, size):
    pad_lu = (img.shape[0] - size) // 2
    pad_rd = img.shape[0] - size - pad_lu
    img = img[pad_lu:-pad_rd, pad_lu:-pad_rd]

    return img


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_vertical(rle_mask):
    if rle_mask is np.nan:
        return False

    if not rle_mask:
        return False

    rle_mask = rle_mask.split(' ')
    for i in range(len(rle_mask)//2):
        if not int(rle_mask[2*i+1]) % 101 == 100:
            return False
    return True
