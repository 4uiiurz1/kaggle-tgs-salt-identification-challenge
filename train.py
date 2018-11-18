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

from sklearn.model_selection import KFold, StratifiedKFold
from skimage.io import imread

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
from metrics import dice_coef, batch_iou
import losses
from utils import str2bool, count_params, crop
from scheduler import CyclicLR


arch_names = archs.__dict__.keys()
loss_names = losses.__dict__.keys()


pixel_criterion = losses.DSVLovaszHingeLoss().cuda()
image_criterion = nn.BCEWithLogitsLoss().cuda()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='SCSECatResUNet34',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: SCSECatResUNet34)')
    parser.add_argument('--pretrained', default=True, type=str2bool,
                        help='use pre-trained model')
    parser.add_argument('--act-func', default='ELUp1',
                        choices=['ReLU', 'ELUp1'])
    parser.add_argument('--first-stride', default=1, type=int)
    parser.add_argument('--loss', default='LovaszHingeLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: LovaszHingeLoss)')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['MultiStepLR', 'StepLR', 'CyclicLR', 'CosineAnnealingLR'],
                        help='scheduler: ' +
                            ' | '.join(['MultiStepLR', 'StepLR', 'CyclicLR', 'CosineAnnealingLR']) +
                            ' (default: CosineAnnealingLR)')
    parser.add_argument('--epochs', default=250, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--reduce-epoch', default=None, type=int, metavar='N',
                        help='epoch for reducing learning rate')
    parser.add_argument('--min-lr', default=1e-4, type=float,
                        help='minimum learning rate')
    parser.add_argument('--early-stop', default=None, type=int,
                        metavar='N', help='early stopping (default: None)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--img-size', default=128, type=int,
                        help='input image size (default: 128)')
    parser.add_argument('--pad', default=False, type=str2bool)
    parser.add_argument('--aug', default=3, type=int,
                        help='augmentation type (default: 3)')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--depth', default=True, type=str2bool)
    parser.add_argument('--coord_conv', default=False, type=str2bool)
    parser.add_argument('--freeze-bn', dest='freeze_bn', action='store_true',
                        help='freeze BatchNorm layers of encoder')
    parser.add_argument('--cv', default='KFold',
                        choices=['KFold', 'Cov'],
                        help='cv: ' +
                            ' | '.join(['KFold', 'Cov']) +
                            ' (default: KFold)')
    parser.add_argument('--n-splits', default=5, type=int)
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use')

    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if args.scheduler == 'CyclicLR':
            scheduler.batch_step()

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        #dice = dice_coef(output, target)
        iou = batch_iou(output, target)
        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'IoU {iou.val:.3f} ({iou.avg:.3f})'.format(
                   epoch, i, len(train_loader), loss=losses, iou=ious))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            #dice = dice_coef(output, target)
            iou = batch_iou(output, target)
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

            if i % args.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'IoU {iou.val:.3f} ({iou.avg:.3f})'.format(
                       i, len(val_loader), loss=losses, iou=ious))

        print(' * Loss {loss.avg:.4f} IoU {iou.avg:.3f}'
              .format(loss=losses, iou=ious))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def main():
    args = parse_args()

    # add model name to args
    if args.name is None:
        args.name = '%s_%s' %(args.arch, datetime.now().strftime('%m%d%H%M'))
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    else:
        criterion = losses.__dict__[args.loss]().cuda(args.gpu)

    cudnn.benchmark = True

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

    for fold, (train_idx, val_idx) in enumerate(cv):
        print('Fold [%d/%d]' %(fold+1, args.n_splits))

        # create model
        print("=> creating model %s (pretrained=%s)" %(args.arch, str(args.pretrained)))
        model = archs.__dict__[args.arch](args)
        if args.freeze_bn:
            model.freeze_bn()

        if args.gpu is not None:
            model = model.cuda(args.gpu)
        else:
            model = torch.nn.DataParallel(model).cuda()

        print(count_params(model))

        if args.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

        if args.scheduler == 'MultiStepLR':
            if args.reduce_epoch is None:
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
            else:
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.reduce_epoch], gamma=0.1)
        elif args.scheduler == 'CyclicLR':
            scheduler = CyclicLR(optimizer, step_size=800)
        elif args.scheduler == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        elif args.scheduler == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.5)

        train_img_paths, val_img_paths = img_paths[train_idx], img_paths[val_idx]
        train_mask_paths, val_mask_paths = mask_paths[train_idx], mask_paths[val_idx]

        train_dataset = Dataset(args, train_img_paths, train_mask_paths)
        val_dataset = Dataset(args, val_img_paths, val_mask_paths, False)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        log = pd.DataFrame(index=[], columns=[
            'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
        ])

        best_loss = float('inf')
        trigger = 0
        for epoch in range(args.epochs):

            if args.scheduler == 'CyclicLR':
                # train for one epoch
                train_log = train(args, train_loader, model, criterion, optimizer, epoch, scheduler)
            else:
                scheduler.step()
                # train for one epoch
                train_log = train(args, train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            val_log = validate(args, val_loader, model, criterion)

            tmp = pd.Series([
                epoch,
                scheduler.get_lr()[0],
                train_log['loss'],
                train_log['iou'],
                val_log['loss'],
                val_log['iou'],
            ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

            log = log.append(tmp, ignore_index=True)
            log.to_csv('models/%s/log_%d.csv' %(args.name, fold+1), index=False)

            trigger += 1

            if val_log['loss'] < best_loss:
                torch.save(model.state_dict(), 'models/%s/model_%d.pth' %(args.name, fold+1))
                best_loss = val_log['loss']
                print("=> saved best model")
                trigger = 0

            # early stopping
            if not args.early_stop is None:
                if epoch > args.epochs * 0.8 and trigger >= args.early_stop:
                    print("=> early stopping")
                    break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
