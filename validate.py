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

from sklearn.model_selection import KFold, StratifiedKFold
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
from metrics import mean_iou, dice_coef
from utils import depth_encode, coord_conv, pad, crop


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    val_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %val_args.name)

    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

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
    train_df = pd.read_csv('input/train.csv')
    img_paths = 'input/train/images/' + train_df['id'].values + '.png'
    mask_paths = 'input/train/masks/' + train_df['id'].values + '.png'

    if args.cv == 'KFold':
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=41)
        cv = kf.split(img_paths)
    elif args.cv == 'Cov':
        train_df['cov'] = 0
        for i in tqdm(range(len(train_df))):
            mask = imread('input/train/masks/' + train_df['id'][i] + '.png')
            mask = mask.astype('float32') / 255
            train_df.loc[i, 'cov'] = ((np.sum(mask>0.5) / 101**2) * 10).astype('int')
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=41)
        cv = skf.split(img_paths, train_df['cov'])

    if not os.path.exists('output/%s/val' %args.name):
        os.makedirs('output/%s/val' %args.name)
    for fold, (train_idx, val_idx) in enumerate(cv):
        print('Fold [%d/%d]' %(fold+1, args.n_splits))

        model.load_state_dict(torch.load('models/%s/model_%d.pth' %(args.name, fold+1)))
        model.eval()

        train_img_paths, val_img_paths = img_paths[train_idx], img_paths[val_idx]
        train_mask_paths, val_mask_paths = mask_paths[train_idx], mask_paths[val_idx]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            for i in tqdm(range(len(val_img_paths))):
                image = imread(val_img_paths[i])
                mask = imread(val_mask_paths[i])

                image = image.astype('float32') / 255
                mask = mask.astype('float32') / 65535

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

                imsave('output/%s/val/%s' %(args.name, os.path.basename(val_img_paths[i])),
                        (pb*255).astype('uint8'))

        torch.cuda.empty_cache()

    # Loss
    losses = []
    for i in tqdm(range(len(mask_paths))):
        mask = imread(mask_paths[i])
        pb = imread('output/%s/val/%s' %(args.name, os.path.basename(img_paths[i])))

        mask = (mask > 65535/2).astype('int')
        pb = pb.astype('float64') / 255

        loss = log_loss(mask.flatten(), pb.flatten(), labels=[0, 1])
        losses.append(loss)

    # IoU
    thrs = np.linspace(0.4, 0.6, 21)
    ious = []
    for thr in thrs:
        print('thr=%0.2f: ' %thr, end='')

        tmp_ious = []
        for i in tqdm(range(len(mask_paths))):
            mask = imread(mask_paths[i])
            pb = imread('output/%s/val/%s' %(args.name, os.path.basename(img_paths[i])))

            mask = (mask > 65535/2).astype('int')
            pb = pb.astype('float64') / 255

            iou = mean_iou(mask, pb>thr)
            tmp_ious.append(iou)
        ious.append(np.mean(tmp_ious))
        print(np.mean(tmp_ious))

    val_info = {
        'loss': np.mean(losses),
        'best_iou': np.max(ious),
        'best_thr': thrs[np.argmax(ious)]
    }

    print('Result -----')
    print('Loss: %f' %val_info['loss']),
    print('Best IoU: %f' %val_info['best_iou']),
    print('Best threshold: %f' %val_info['best_thr'])
    print('------------')

    with open('models/%s/val_info.txt' %args.name, 'w') as f:
        print('Result -----', file=f)
        print('Loss: %f' %val_info['loss'], file=f),
        print('Best IoU: %f' %val_info['best_iou'], file=f),
        print('Best threshold: %.2f' %val_info['best_thr'], file=f)
        print('------------', file=f)

    joblib.dump(val_info, 'models/%s/val_info.pkl' %args.name)


if __name__ == '__main__':
    main()
