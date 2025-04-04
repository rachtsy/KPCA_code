# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
from fvcore.nn import FlopCountAnalysis
import wandb
import numpy as np


import torch


from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import pdb

def train_one_epoch(reconstruct_samples, model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,wandb=False):
    # put our model in training mode... so that drop out and batch normalisation does not affect it
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    print(reconstruct_samples)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if reconstruct_samples == None and i==2:
            print("assigning reconstruction samples idx")
            reconstruct_samples = samples
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # if i == 50:
        #     break
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)


        with torch.cuda.amp.autocast():
            # flops = FlopCountAnalysis(model,samples)
            # print(flops.total()/1e9)
            # assert 1==2
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
       
        # break

        # loss is a tensor, averaged over the mini batch
        loss_value = loss.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            f = open("error.txt", "a")
            # writing in the file
            f.write("Loss is {}, stopping training".format(loss_value))
            # closing the file
            f.close() 
            sys.exit(1)


        optimizer.zero_grad()


        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # provides optimisation step for model
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)


        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)


        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if wandb:
        for k, meter in metric_logger.meters.items():
            wandb.log({k: meter.global_avg, 'epoch': epoch})
    model.module.reconstruct = True
    reconstruct_samples = reconstruct_samples.to(device, non_blocking=True)
    outputs = model(reconstruct_samples)
    model.module.reconstruct = False
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, reconstruct_samples

# evaluate on 1000 images in imagenet/val folder
@torch.no_grad()
def evaluate(reconstruct_samples, data_loader, model, device, attn_only=False, batch_limit=0,epoch=0,wandb=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    model.module.test()
    # i = 0
    if not isinstance(batch_limit, int) or batch_limit < 0:
        batch_limit = 0
    attn = []
    pi = []
    for i, (images, target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if reconstruct_samples == None:
            print("assigning reconstruction samples idx")
            reconstruct_samples = images
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        with torch.cuda.amp.autocast():
            if attn_only:
                output, _aux = model(images)
                attn.append(_aux[0].detach().cpu().numpy())
                pi.append(_aux[1].detach().cpu().numpy())
                del _aux
            else:
                output = model(images)
            loss = criterion(output, target)

        # print(output.shape,target.shape)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        r = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    if wandb:
        for k, meter in metric_logger.meters.items():
            wandb.log({f'test_{k}': meter.global_avg , 'epoch':epoch})

    model.module.reconstruct = True
    model.module.test_check = True
    reconstruct_samples = reconstruct_samples.to(device, non_blocking=True)
    outputs = model(reconstruct_samples)
    model.module.reconstruct = False
    model.module.test_check = False
    if attn_only:
        return r, (attn, pi)
    return r, reconstruct_samples


