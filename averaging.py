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
from metrics import mean_iou, dice_coef
from utils import rle_encode


def main():
    weights = {
        'SCSECatResUNet34_pseudo': 0.4
        'SCSECatSEResNeXt32x4dUNet50_pseudo': 0.3
        'SCSECatResUNet34_pseudo_pad': 0.3
    }

    avg_name = 'averaging_' + datetime.now().strftime('%m%d%H%M')

    if not os.path.exists('models/%s' %avg_name):
        os.makedirs('models/%s' %avg_name)

    print('Weights -----')
    for model_name in weights.keys():
        print('%s: %f' %(model_name, weights[model_name]))
    print('------------')

    with open('models/%s/weights.txt' %avg_name, 'w') as f:
        print('Weights -----', file=f)
        for model_name in weights.keys():
            print('%s: %f' %(model_name, weights[model_name]), file=f)
        print('------------', file=f)

    # Data loading code
    train_df = pd.read_csv('input/train.csv')
    img_paths = 'input/train/images/' + train_df['id'].values + '.png'
    mask_paths = 'input/train/masks/' + train_df['id'].values + '.png'

    if not os.path.exists('output/%s/val' %avg_name):
        os.makedirs('output/%s/val' %avg_name)

    # Loss
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        losses = []
        for i in tqdm(range(len(mask_paths))):
            mask = imread(mask_paths[i])
            pb = np.zeros(mask.shape)
            for model_name in weights.keys():
                pb_ = imread('output/%s/val/%s' %(model_name, os.path.basename(img_paths[i])))
                pb += weights[model_name] * pb_

            mask = (mask > 65535/2).astype('int')
            pb = pb.astype('float64') / 255

            imsave('output/%s/val/%s' %(avg_name, os.path.basename(img_paths[i])),
                    (pb*255).astype('uint8'))

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
            pb = np.zeros(mask.shape)
            for model_name in weights.keys():
                pb_ = imread('output/%s/val/%s' %(model_name, os.path.basename(img_paths[i])))
                pb += weights[model_name] * pb_

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

    with open('models/%s/val_info.txt' %avg_name, 'w') as f:
        print('Result -----', file=f)
        print('Loss: %f' %val_info['loss'], file=f),
        print('Best IoU: %f' %val_info['best_iou'], file=f),
        print('Best threshold: %f' %val_info['best_thr'], file=f)
        print('------------', file=f)

    joblib.dump(val_info, 'models/%s/val_info.pkl' %avg_name)

    # Data loading code
    submission = pd.read_csv('input/sample_submission.csv')
    img_paths = 'input/test/images/' + submission['id'].values + '.png'

    if not os.path.exists('output/%s/test' %avg_name):
        os.makedirs('output/%s/test' %avg_name)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for i in tqdm(range(len(submission))):
            img_id = submission['id'][i]

            pb = np.zeros(mask.shape)
            for model_name in weights.keys():
                pb_ = imread('output/%s/test/%s' %(model_name, img_id+'.png'))
                pb += weights[model_name] * pb_
            pb = pb.astype('float32') / 255

            imsave('output/%s/test/%s' %(avg_name, img_id+'.png'),
                    (pb*255).astype('uint8'))

            submission['rle_mask'][i] = rle_encode(pb>val_info['best_thr'])

    submission.to_csv('submission/%s.csv.gz' %avg_name, compression='gzip', index=False)


if __name__ == '__main__':
    main()
