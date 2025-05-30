from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader
from network import Network
from dataloader import VOC
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from tensorboardX import SummaryWriter

try:
    from azureml.core import Run

    azure = True
    run = Run.get_context()
except:
    azure = False

parser = argparse.ArgumentParser()

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False

with Engine(custom_parser=parser) as engine:
    cudnn.benchmark = True

    seed = config.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Seed all GPUs for reproducibility if using multiple

    train_loader, train_sampler = get_train_loader(engine, VOC, train_source=config.train_source, \
                                                   unsupervised=False)
    unsupervised_train_loader, unsupervised_train_sampler = get_train_loader(engine, VOC, \
                                                                             train_source=config.unsup_source,
                                                                             unsupervised=True)

    tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    generate_tb_dir = config.tb_dir + '/tb'
    logger = SummaryWriter(log_dir=tb_dir)
    engine.link_tb(tb_dir, generate_tb_dir)

    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    criterion_csst = nn.MSELoss(reduction='mean')

    BatchNorm2d = nn.BatchNorm2d

    model = Network(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)
    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    base_lr = config.lr

    params_list_l = []
    params_list_l = group_weight(params_list_l, model.branch1.backbone,
                                 BatchNorm2d, base_lr)
    for module in model.branch1.business_layer:
        params_list_l = group_weight(params_list_l, module, BatchNorm2d,
                                     base_lr)

    optimizer_l = torch.optim.SGD(params_list_l,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    params_list_r = []
    params_list_r = group_weight(params_list_r, model.branch2.backbone,
                                 BatchNorm2d, base_lr)
    for module in model.branch2.business_layer:
        params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                     base_lr)

    optimizer_r = torch.optim.SGD(params_list_r,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    # Move model to the primary device first
    model.to(engine.device)
    print(f"Model initially moved to {engine.device}")

    # If multiple GPUs are specified in engine.devices, wrap with DataParallel
    if engine.device.type == 'cuda' and len(engine.devices) > 1:
        print(f"Wrapping model with DataParallel for GPUs: {engine.devices}")
        model = torch.nn.DataParallel(model, device_ids=engine.devices)
        # The DataParallel wrapper itself should reside on the primary device,
        # and it handles placing replicas. model.to(engine.device) before wrapping
        # or ensuring DataParallel itself handles this is usually sufficient.
        # Calling model.to(engine.device) again on the DP-wrapped model is fine.
        model.to(engine.device)
        print(f"DataParallel model output device: {engine.device} (typically gathers here)")
    elif engine.device.type == 'cuda':
        print(f"Using single GPU: {engine.device}")
    else:
        print("Using CPU for model.")

    print("help help 0")

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer_l=optimizer_l, optimizer_r=optimizer_r)

    print("help help 1")

    if engine.continue_state_object:
        engine.restore_checkpoint()

    print("help help 2")

    # Set model to training mode. If DataParallel, this applies to model.module
    model.train()
    print('begin train')

    for epoch in range(engine.state.epoch, config.nepochs):
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        dataloader = iter(train_loader)
        unsupervised_dataloader = iter(unsupervised_train_loader)

        sum_loss_sup = 0
        sum_loss_sup_r = 0
        sum_cps = 0

        for idx in pbar:
            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            unsup_minibatch = next(unsupervised_dataloader)

            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs = unsup_minibatch['data']

            is_cuda_primary = engine.device.type == 'cuda'
            # Move data to the primary device. DataParallel will scatter it.
            imgs = imgs.to(engine.device, non_blocking=is_cuda_primary)
            unsup_imgs = unsup_imgs.to(engine.device, non_blocking=is_cuda_primary)
            gts = gts.to(engine.device, non_blocking=is_cuda_primary)

            b, c, h, w = imgs.shape
            # model() call is handled correctly by DataParallel
            _, pred_sup_l = model(imgs, step=1)
            _, pred_unsup_l = model(unsup_imgs, step=1)
            _, pred_sup_r = model(imgs, step=2)
            _, pred_unsup_r = model(unsup_imgs, step=2)

            pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
            pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
            _, max_l = torch.max(pred_l, dim=1)
            _, max_r = torch.max(pred_r, dim=1)
            max_l = max_l.long()
            max_r = max_r.long()
            cps_loss = criterion(pred_l, max_r) + criterion(pred_r, max_l)
            cps_loss = cps_loss * config.cps_weight

            loss_sup = criterion(pred_sup_l, gts)
            loss_sup_r = criterion(pred_sup_r, gts)

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer_l.param_groups[0]['lr'] = lr
            optimizer_l.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_l.param_groups)):
                optimizer_l.param_groups[i]['lr'] = lr
            optimizer_r.param_groups[0]['lr'] = lr
            optimizer_r.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_r.param_groups)):
                optimizer_r.param_groups[i]['lr'] = lr

            loss = loss_sup + loss_sup_r + cps_loss
            loss.backward()  # Gradients are averaged across GPUs by DataParallel
            optimizer_l.step()
            optimizer_r.step()

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % loss_sup.item() \
                        + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                        + ' loss_cps=%.4f' % cps_loss.item()

            sum_loss_sup += loss_sup.item()
            sum_loss_sup_r += loss_sup_r.item()
            sum_cps += cps_loss.item()
            pbar.set_description(print_str, refresh=False)

        logger.add_scalar('train_loss_sup', sum_loss_sup / len(pbar), epoch)
        logger.add_scalar('train_loss_sup_r', sum_loss_sup_r / len(pbar), epoch)
        logger.add_scalar('train_loss_cps', sum_cps / len(pbar), epoch)

        if azure:
            run.log(name='Supervised Training Loss', value=sum_loss_sup / len(pbar))
            run.log(name='Supervised Training Loss right', value=sum_loss_sup_r / len(pbar))
            run.log(name='Supervised Training Loss CPS', value=sum_cps / len(pbar))

        if (epoch > config.nepochs // 2) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            engine.save_and_link_checkpoint(config.snapshot_dir,
                                            config.log_dir,
                                            config.log_dir_link)