# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import joblib
import cv2

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import archs
from utils import rle_encode
from metrics import mean_iou, dice_coef
from utils import depth_encode, coord_conv, pad, crop


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    test_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %test_args.name)
    val_info = joblib.load('models/%s/val_info.pkl' %args.name)

    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    print('Result -----')
    print('Loss: %f' %val_info['loss']),
    print('Best IoU: %f' %val_info['best_iou']),
    print('Best threshold: %f' %val_info['best_thr'])
    print('------------')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # create model
    print("=> creating model %s" %args.arch)
    model = archs.__dict__[args.arch](args)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Data loading code
    submission = pd.read_csv('input/sample_submission.csv')
    img_paths = 'input/test/images/' + submission['id'].values + '.png'

    if not os.path.exists('output/%s/test' %args.name):
        os.makedirs('output/%s/test' %args.name)
    for i in range(args.n_splits):
        print('Fold [%d/%d]' %(i+1, args.n_splits))

        model.load_state_dict(torch.load('models/%s/model_%d.pth' %(args.name, i+1)))
        model.eval()

        if not os.path.exists('output/%s/test/%d' %(args.name, i+1)):
            os.makedirs('output/%s/test/%d' %(args.name, i+1))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            for j in tqdm(range(len(img_paths))):
                image = imread(img_paths[j])
                image = image.astype('float32') / 255

                if 'Res' in args.arch:
                    means = [0.485, 0.456, 0.406]
                    stds = [0.229, 0.224, 0.225]
                    for c in range(3):
                        image[:,:,c] = (image[:,:,c] - means[c]) / stds[c]

                if 'Inception' in args.arch:
                    means = [0.5, 0.5, 0.5]
                    stds = [0.5, 0.5, 0.5]
                    for c in range(3):
                        image[:,:,c] = (image[:,:,c] - means[c]) / stds[c]

                pbs = []

                if not args.pad:
                    input = cv2.resize(image, (args.img_size, args.img_size))
                else:
                    input = pad(image, args.img_size)
                if args.depth:
                    input = depth_encode(input)
                if args.coord_conv:
                    input = coord_conv(input)
                input = input.transpose((2, 0, 1))

                input = input[np.newaxis, :, :, :]
                input = torch.from_numpy(input)
                input = input.cuda(args.gpu)

                output = model(input)
                output = F.sigmoid(output)

                pb = output.data.cpu().numpy()
                pb = pb[0, 0, :, :]
                if not args.pad:
                    pb = cv2.resize(pb, (101, 101))
                else:
                    pb = crop(pb, 101)
                pbs.append(pb)

                if not args.pad:
                    input = cv2.resize(image[:, ::-1, :], (args.img_size, args.img_size))
                else:
                    input = pad(image[:, ::-1, :], args.img_size)
                if args.depth:
                    input = depth_encode(input)
                if args.coord_conv:
                    input = coord_conv(input)
                input = input.transpose((2, 0, 1))

                input = input[np.newaxis, :, :, :]
                input = torch.from_numpy(input)
                input = input.cuda(args.gpu)

                output = model(input)
                output = F.sigmoid(output)

                pb = output.data.cpu().numpy()[:, :, :, ::-1]
                pb = pb[0, 0, :, :]
                if not args.pad:
                    pb = cv2.resize(pb, (101, 101))
                else:
                    pb = crop(pb, 101)
                pbs.append(pb)

                pb = np.mean(pbs, axis=0)

                imsave('output/%s/test/%d/%s' %(args.name, i+1, os.path.basename(img_paths[j])),
                        (pb*255).astype('uint8'))
        torch.cuda.empty_cache()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for i in tqdm(range(len(submission))):
            img_id = submission['id'][i]

            pbs = []
            for j in range(args.n_splits):
                pb = imread('output/%s/test/%d/%s' %(args.name, j+1, img_id+'.png'))
                pb = pb.astype('float32') / 255
                pbs.append(pb)

            pb = np.mean(pbs, axis=0)

            imsave('output/%s/test/%s' %(args.name, img_id+'.png'),
                    (pb*255).astype('uint8'))

            submission['rle_mask'][i] = rle_encode(pb>val_info['best_thr'])

    submission.to_csv('submission/%s.csv.gz' %args.name, compression='gzip', index=False)


if __name__ == '__main__':
    main()
