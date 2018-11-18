import random
import math
import pandas as pd
import os
import cv2

import numpy as np

from skimage.io import imread

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy import linalg
import scipy.ndimage as ndi


def random_fliplr(img, mask, prob=0.5):
    if random.uniform(0, 1) > prob:
        img = img[:, ::-1, :].copy()
        mask = mask[:, ::-1].copy()

    return img, mask


def random_erase(img, mean=None, prob=0.5, sl=0.02, sh=0.4, r=0.3, type=1):
    if random.uniform(0, 1) < prob:
        return img

    while True:
        area = random.uniform(sl, sh) * img.shape[0] * img.shape[1]
        ratio = random.uniform(r, 1/r)

        h = int(round(math.sqrt(area * ratio)))
        w = int(round(math.sqrt(area / ratio)))

        if h < img.shape[0] and w < img.shape[1]:
            x = random.randint(0, img.shape[0] - h)
            y = random.randint(0, img.shape[1] - w)
            if mean is None:
                for c in range(3):
                    if type == 1:
                        img[x:x+h, y:y+w, c] = random.uniform(0, 1)
                    elif type == 2:
                        img[x:x+h, y:y+w, c] = np.random.rand(h, w)
            else:
                for c in range(3):
                    img[x:x+h, y:y+w, c] = mean[c]
            return img


def random_shift(X, y, wrg, hrg, row_axis=0, col_axis=1, channel_axis=2,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Shifted Numpy image tensor.
    """
    h, w = X.shape[row_axis], X.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    X = apply_transform(X, transform_matrix, channel_axis, fill_mode, cval)
    y = apply_transform(y[:,:,np.newaxis], transform_matrix, channel_axis, fill_mode, cval)[:,:,0]

    return X, y


def random_adjgamma(img, min=0.7, max=1.2):
    gamma = random.uniform(min, max)
    img = adjust_gamma(img, gamma)

    return img


"""-------------------------------------------------------------------------------------------------"""


def random_rot90(img, mask):
    k = random.randrange(4)
    img = np.rot90(img, k=k).copy()
    mask = np.rot90(mask, k=k).copy()

    return img, mask


def random_flip(img, mask, prob=0.5):
    if random.uniform(0, 1) > prob:
        img = img[:, ::-1, :].copy()
        mask = mask[:, ::-1].copy()

    if random.uniform(0, 1) > prob:
        img = img[::-1, :, :].copy()
        mask = mask[::-1, :].copy()

    return img, mask


def random_shiftlr(img, mask, wrg=0.2, prob=0.5):
    if random.uniform(0, 1) > prob:
        r = random.randint(-int(img.shape[1] * wrg), int(img.shape[1] * wrg))
        if not r == 0:
            tmp = np.zeros(img.shape)
            tmp[:, :r] = img[:, :r]
            img[:, :-r] = img[:, r:]
            img[:, -r:] = tmp[:, :r]
            tmp = np.zeros(mask.shape)
            tmp[:, :r] = mask[:, :r]
            mask[:, :-r] = mask[:, r:]
            mask[:, -r:] = tmp[:, :r]

    return img, mask


def elastic_distortion(img, mask, alpha=1000, sigma=30):
    dx = gaussian_filter((np.random.rand(img.shape[0], img.shape[1]) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(img.shape[0], img.shape[1]) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    for i in range(img.shape[2]):
        img[:,:,i] = map_coordinates(img[:,:,i], indices, order=1, mode='nearest').reshape(img.shape[:2])
    mask = map_coordinates(mask, indices, order=1, mode='nearest').reshape(img.shape[:2])

    return img, mask


def mixup(img, mask, img_paths, mask_paths, alpha=0.2, prob=0.5):
    if random.uniform(0, 1) < prob:
        return img, mask

    idx = random.randrange(len(img_paths))
    img2 = imread(img_paths[idx])
    mask2 = imread(mask_paths[idx])

    img2 = img2.astype('float32') / 255
    mask2 = mask2.astype('float32') / 65535

    l = np.random.beta(alpha, alpha)
    img = img*l + img2*(1-l)
    mask = mask*l + mask2*(1-l)

    return img, mask


def random_gaussian_blur(img, prob=0.5, sl=1, sh=3):
    if random.uniform(0, 1) > prob:
        for c in range(3):
            img[:, :, c] = gaussian_filter(img[:, :, c], random.uniform(sl, sh))
        return img

    else:
        return img


def random_gaussian_noise(img, prob=0.5, s=0.03):
    if random.uniform(0, 1) > prob:
        noise = np.tile(np.random.normal(0, s, (101, 101, 1)), (1, 1, 3))
        img += noise
        return img

    else:
        return img


def random_zoom(X, y, zoom_range, row_axis=0, col_axis=1, channel_axis=2,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Zoomed Numpy image tensor.
    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = X.shape[row_axis], X.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    X = apply_transform(X, transform_matrix, channel_axis, fill_mode, cval)
    y = apply_transform(y[:,:,np.newaxis], transform_matrix, channel_axis, fill_mode, cval)[:,:,0]

    return X, y


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def adjust_gamma(img, gamma=1, gain=1):
    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number.")

    img = (img ** gamma) * gain
    return img


def random_shift_puzzle(args, img, mask, img_path, thr=0.5):
    img_id = os.path.splitext(os.path.basename(img_path))[0]

    dirs = []
    if df.loc[df.id == img_id, 'left_score'].values[0] > thr:
        dirs.append('left')
    if df.loc[df.id == img_id, 'right_score'].values[0] > thr:
        dirs.append('right')
    if df.loc[df.id == img_id, 'up_score'].values[0] > thr:
        dirs.append('up')
    if df.loc[df.id == img_id, 'down_score'].values[0] > thr:
        dirs.append('down')

    if not dirs:
        return img, mask

    dir = random.choice(dirs)
    img2_path = df.loc[df.id == img_id, dir].values[0]
    if 'train' in img2_path:
        mask2_path = 'input/train/masks/' + os.path.basename(img2_path)
    else:
        mask2_path = 'output/%s/test/' %args.org_name + os.path.basename(img2_path)

    img2 = imread(img2_path)
    mask2 = imread(mask2_path)

    img2 = img2.astype('float32') / 255
    if 'train' in img2_path:
        mask2= (mask2>65535/2).astype('float32')
    else:
        if args.loss == 'LovaszHingeLoss':
            mask2 = (mask2>127).astype('float32')
        else:
            mask2 = mask2.astype('float32') / 255

    if not args.pad:
        img2 = cv2.resize(img2, (args.img_size, args.img_size))
        mask2 = cv2.resize(mask2, (args.img_size, args.img_size))
    else:
        img2 = pad(img2, args.img_size)
        mask2 = pad(mask2, args.img_size)

    if dir == 'left':
        cat_img = np.concatenate([img2, img], axis=1)
        cat_mask = np.concatenate([mask2, mask], axis=1)
    elif dir == 'right':
        cat_img = np.concatenate([img, img2], axis=1)
        cat_mask = np.concatenate([mask, mask2], axis=1)
    elif dir == 'up':
        cat_img = np.concatenate([img2, img], axis=0)
        cat_mask = np.concatenate([mask2, mask], axis=0)
    else:
        cat_img = np.concatenate([img, img2], axis=0)
        cat_mask = np.concatenate([mask, mask2], axis=0)

    if cat_img.shape[0] == args.img_size:
        h = 0
        w = random.randint(1, cat_img.shape[1] - args.img_size)
    else:
        h = random.randint(0, cat_img.shape[0] - args.img_size)
        w = 0

    img = cat_img[h:h+args.img_size, w:w+args.img_size]
    mask = cat_mask[h:h+args.img_size, w:w+args.img_size]

    return img, mask
