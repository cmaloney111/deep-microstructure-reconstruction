#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import utils
import models.builder as builder
import dataloader

def parse_args():
    print('Parsing argumments')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg19', type=str, help='backbone architechture')
    parser.add_argument('--dataset-name', type=str, help="name of dataset to use for training")
    parser.add_argument('--checkpoint', type=str, help="checkpoint to use if resuming training")
    parser.add_argument('--workers', default=16, type=int, metavar='N', 
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (default 100)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual start epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N',
                        help='mini-batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='N', help='initial learning rate (default: 0.1)', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='N', help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='N', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', 
                        help='print frequency (default: 10)')
    parser.add_argument('--pth-save-fold', default='results', type=str,
                        help='The folder to save pth files (default: results)')
    parser.add_argument('--pth-save-epoch', default=10, type=int, metavar='N',
                        help='The interval in epoch with which to save the pth files (default: 10)')
    parser.add_argument('--parallel', type=int, default=0, metavar='N',
                        help='1 for parallel, 0 for non-parallel (default: 0)')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training (default: tcp://localhost:10001)')                                            

    args = parser.parse_args()
    return args

def main(args):
    ngpus_per_node = torch.cuda.device_count()
    print('Number of gpus: {}'.format(ngpus_per_node))

    args.model_save_folder = os.path.join(args.pth_save_fold, args.arch)
    if not os.path.exists(args.model_save_folder):
        os.makedirs(args.model_save_folder)

    if args.parallel == 1:        
        args.gpus = ngpus_per_node
        args.nodes = 1
        args.nr = 0
        args.world_size = args.gpus * args.nodes
        args.workers = int(args.workers / args.world_size)
        args.batch_size = int(args.batch_size / args.world_size)
        mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
    else:
        args.world_size = 1
        main_worker(ngpus_per_node, args)
    
def main_worker(gpu, args):
    utils.init_seeds(1 + gpu, cuda_deterministic=False)
    if args.parallel == 1:
        args.gpu = gpu
        args.rank = args.nr * args.gpus + args.gpu
        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)  
    else:
        args.rank = 0
        args.gpus = 1 
    if args.rank == 0:
        print('Modeling network: {}'.format(args.arch))
    model = builder.BuildAutoEncoder(args) 
    if args.checkpoint:
        print('Loading checkpoint: {}'.format(args.checkpoint))
        utils.load_dict(args.checkpoint, model)

    if args.rank == 0:       
        total_params = sum(p.numel() for p in model.parameters())
        print('Number of parameters: {}'.format(total_params))
    
    if args.rank == 0:
        print('Building optimizer: sgd')
    # could use adam
    optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)    
    
    if args.rank == 0:
        print('Building dataloader and criterion')
    train_loader = dataloader.train_loader(args)
    criterion = nn.MSELoss()

    if args.rank == 0:
        print('Beginning training')
    model.train()

    for epoch in range(args.start_epoch, args.epochs):
        global current_lr
        current_lr = utils.adjust_learning_rate_cosine(optimizer, epoch, args)

        if args.parallel == 1:
            train_loader.sampler.set_epoch(epoch)
        
        train(train_loader, model, criterion, optimizer, epoch, args)
        
        if epoch % args.pth_save_epoch == 0 and args.rank == 0:
            state_dict = model.state_dict()
            torch.save({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': state_dict,
                        'optimizer' : optimizer.state_dict(),
                        },
                        os.path.join(args.model_save_folder, '{}.pth'.format(str(epoch).zfill(3))))
            print('Saving pth file for epoch {}'.format(epoch + 1))

def train(train_loader, model, criterion, optimizer, epoch, args):
    load_time = utils.AverageMeter('Time:', ':6.2f')
    losses = utils.AverageMeter('Loss:', ':.4f')
    learning_rate = utils.AverageMeter('LR:', ':.4f')    
    progress = utils.ProgressMeter(
        len(train_loader),
        [load_time, losses, learning_rate],
        prefix="Epoch: [{}]".format(epoch+1))
    end = time.time()
    learning_rate.update(current_lr)

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        losses.update(loss.item(), input.size(0))          
        if args.rank == 0:
            load_time.update(time.time() - end)        
            end = time.time()   
        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)