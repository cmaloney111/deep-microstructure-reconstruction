#!/usr/bin/env python

import os
import time
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import utils
import models.builder as builder
import dataloader

def parse_args():
    print('Parsing argumments')
    parser = argparse.ArgumentParser(description='Evaluate for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, help='backbone architechture')
    parser.add_argument('--workers', default=0, type=int, metavar='N', 
                        help='number of data loading workers (default: 0)')
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N',
                        help='mini-batch size (default: 8)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--checkpoint', type=str, help='checkpoint of model to use (.pth file)')
    parser.add_argument('--folder', type=str, help='folder of images to use for evaluation')   
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual start epoch number (useful on restarts)')                                 
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs (default 100)') 

    args = parser.parse_args()
    args.parallel = 0
    return args

def main(args):
    ngpus_per_node = torch.cuda.device_count()
    print('Number of gpus: {}'.format(ngpus_per_node))
    utils.init_seeds(1, cuda_deterministic=False)
    print('Modeling network: {}'.format(args.arch))

    model = builder.BuildAutoEncoder(args)      
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters: {}'.format(total_params))
    
    print('Building dataloader and criterion')
    val_loader = dataloader.val_loader(args)
    criterion = nn.MSELoss()

    print('Beginning evaluation')
    if args.folder:
        best_loss = None
        best_epoch = 1
        losses = []
        for epoch in range(args.start_epoch, args.epochs):
            print()
            print("Epoch {}".format(epoch+1))
            checkpoint_path = os.path.join(args.folder, "%03d.pth" % epoch)
            print('Loading state dict from {}'.format(checkpoint_path))
            utils.load_dict(checkpoint_path, model)
            loss = do_evaluate(val_loader, model, criterion, args)
            print("Evaluation loss: {:.4f}".format(loss))

            losses.append(loss)
            if best_loss:
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch + 1
            else:
                best_loss = loss
        print()
        print("Best loss: {:.4f} appears in {}".format(best_loss, best_epoch))

        max_loss = max(losses)

        plt.figure(figsize=(7,7))

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.xlim((0,args.epochs+1)) 
        plt.ylim([0, float('%.1g' % (1.22*max_loss))])

        plt.scatter(range(1, args.epochs+1), losses, s=9)

        plt.savefig("figs/evalall.jpg")

    else:
        print('Loading parameters from {}'.format(args.checkpoint))
        utils.load_dict(args.checkpoint, model)
        loss = do_evaluate(val_loader, model, criterion, args)
        print("Evaluation loss: {:.4f}".format(loss))


def do_evaluate(val_loader, model, criterion, args):
    load_time = utils.AverageMeter('Time:', ':6.2f')
    losses = utils.AverageMeter('Loss:', ':.4f')
    
    progress = utils.ProgressMeter(
        len(val_loader),
        [load_time, losses],
        prefix="Evaluate ")
    end = time.time()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(input)
            loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))          
            load_time.update(time.time() - end)        
            end = time.time()   
            if i % args.print_freq == 0:
                progress.display(i)
    
    return losses.avg

if __name__ == '__main__':
    args = parse_args()
    main(args)


