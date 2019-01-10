import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict
from datetime import datetime
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *
from dataset import ClassAwareMNIST
from mixed_context_loss import MixedContextLoss
import archs

arch_names = archs.__dict__.keys()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='BNNet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: BNNet)')
    parser.add_argument('--theta_glo', default=1.15, type=float,
                        help='theta_glo')
    parser.add_argument('--delta', default=5, type=int,
                        help='delta')
    parser.add_argument('--gamma', default=0.5, type=float,
                        help='gamma')
    parser.add_argument('--scale-aware', default=True, type=str2bool,
                        help='scale aware')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min-lr', default=1e-3, type=float,
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.5, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    for i, (input1, input2, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input1 = input1.cuda()
        input2 = input2.cuda()

        output1 = model(input1)
        output2 = model(input2)

        loss = criterion(output1, output2, target)

        losses.update(loss.item(), input1.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input1, input2, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input1 = input1.cuda()
            input2 = input2.cuda()

            output1 = model(input1)
            output2 = model(input2)

            loss = criterion(output1, output2, target)

            losses.update(loss.item(), input1.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
    ])

    return log


def main():
    args = parse_args()

    if args.name is None:
        args.name = '%s' %args.arch
        args.name += '_gamma%.1f' %args.gamma
        if args.scale_aware:
            args.name += '_scale_aware'

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

    criterion = MixedContextLoss(args.theta_glo, args.delta, args.gamma, args.scale_aware).cuda()

    cudnn.benchmark = True

    # data loading code
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = ClassAwareMNIST(
        train=True,
        transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8)

    test_set = ClassAwareMNIST(
        train=False,
        transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8)

    # create model
    model = archs.__dict__[args.arch]()
    model = model.cuda()

    print(model)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'val_loss'
    ])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(args.min_lr/args.lr)**(1/(args.epochs-1)))

    best_loss = float('inf')
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))
        scheduler.step()

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, test_loader, model, criterion)

        print('loss %.4f - val_loss %.4f'
            %(train_log['loss'], val_log['loss']))

        tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            train_log['loss'],
            val_log['loss'],
        ], index=['epoch', 'lr', 'loss', 'val_loss'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)


if __name__ == '__main__':
    main()
