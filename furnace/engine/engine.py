#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/8/2 下午3:23
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : engine.py
import os
import os.path as osp
import time
import argparse

import shutil
import torch

from .logger import get_logger
from .version import __version__
from utils.pyt_utils import load_model, parse_devices, extant_file, link_file, \
    ensure_dir

logger = get_logger()


class State(object):
    def __init__(self):
        self.epoch = 0
        self.iteration = 0
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.optimizer_l = None
        self.optimizer_r = None

    def register(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Engine(object):
    def __init__(self, custom_parser=None):
        self.version = __version__
        logger.info(
            "PyTorch Version {}, Furnace Version {}".format(torch.__version__,
                                                            self.version))
        self.state = State()
        self.device = None  # Will be the primary torch.device object (e.g., cuda:0)
        self.devices = []  # List of GPU device IDs (int), e.g., [0, 1] or [0]
        self.distributed = False  # Always False for single node

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        if self.args.continue_fpath is not None and os.path.exists(self.args.continue_fpath):
            self.continue_state_object = self.args.continue_fpath
        else:
            self.continue_state_object = None
        print('continue_state_object: ', self.continue_state_object)

        # Device setup for single-node multi-GPU / single GPU / CPU
        # self.args.devices will be e.g. "0", "0,1", "" (default), "cpu"
        parsed_gpu_ids = parse_devices(self.args.devices)  # list of ints, or [] for 'cpu'

        if parsed_gpu_ids and torch.cuda.is_available():  # We have GPU IDs and CUDA is truly available
            self.devices = parsed_gpu_ids  # Store all specified GPU IDs
            if not self.devices:  # Should not happen if parsed_gpu_ids is true, but defensive
                logger.warning("Parsed GPU IDs were non-empty but resulted in an empty list. Defaulting to CPU.")
                self.device = torch.device("cpu")
                self.devices = []
                logger.info("Using CPU.")
            else:
                # The primary device will be the first one in the list
                self.device = torch.device(f"cuda:{self.devices[0]}")
                torch.cuda.set_device(self.devices[0])  # Set default CUDA device for PyTorch context
                logger.info(f"Using GPU(s): {self.devices}. Primary device: {self.device}")
        else:
            self.device = torch.device("cpu")
            self.devices = []  # No CUDA GPUs to use
            logger.info("Using CPU. If you intended to use GPU, please check CUDA availability and device argument.")

    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='0',  # Default to GPU 0 if available
                       help='set target device(s) e.g. "0" for cuda:0, "0,1" for cuda:0 and cuda:1, "cpu" for CPU.')
        p.add_argument('-c', '--continue', type=str,
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')
        p.add_argument('--debug', default=0, type=int,
                       help='whether to use the debug mode')

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def update_iteration(self, epoch, iteration):
        self.state.epoch = epoch
        self.state.iteration = iteration

    def save_checkpoint(self, path):
        logger.info("Saving checkpoint to file {}".format(path))
        t_start = time.time()

        state_dict = {}
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        # If model is wrapped by DataParallel, actual model is model.module
        model_to_save = self.state.model.module if isinstance(self.state.model,
                                                              torch.nn.DataParallel) else self.state.model
        model_state_dict = model_to_save.state_dict()

        for k, v in model_state_dict.items():
            # The original code already handles stripping 'module.' if it somehow exists
            # in the raw model's state_dict (which is unusual).
            # DataParallel itself adds 'module.' to keys, so we save model.module.state_dict()
            key = k
            if k.startswith('module.'):  # This condition is unlikely if we save model_to_save.state_dict()
                key = k[7:]
            new_state_dict[key] = v
        state_dict['model'] = new_state_dict

        if self.state.optimizer is not None:
            state_dict['optimizer'] = self.state.optimizer.state_dict()
        if self.state.optimizer_l is not None:
            state_dict['optimizer_l'] = self.state.optimizer_l.state_dict()
        if self.state.optimizer_r is not None:
            state_dict['optimizer_r'] = self.state.optimizer_r.state_dict()
        state_dict['epoch'] = self.state.epoch
        state_dict['iteration'] = self.state.iteration

        t_iobegin = time.time()
        torch.save(state_dict, path)
        del state_dict
        del new_state_dict
        t_end = time.time()
        logger.info(
            "Save checkpoint to file {}, "
            "Time usage:\n\tprepare snapshot: {}, IO: {}".format(
                path, t_iobegin - t_start, t_end - t_iobegin))

    def link_tb(self, source, target):
        ensure_dir(source)
        ensure_dir(target)
        link_file(source, target)

    def save_and_link_checkpoint(self, snapshot_dir, log_dir, log_dir_link, name=None):
        ensure_dir(snapshot_dir)
        if not osp.exists(log_dir_link):
            link_file(log_dir, log_dir_link)
        if name is None:
            current_epoch_checkpoint = osp.join(snapshot_dir, 'epoch-{}.pth'.format(
                self.state.epoch))
        else:
            current_epoch_checkpoint = osp.join(snapshot_dir, '{}.pth'.format(
                name))

        if os.path.exists(current_epoch_checkpoint):
            os.remove(current_epoch_checkpoint)

        self.save_checkpoint(current_epoch_checkpoint)
        last_epoch_checkpoint = osp.join(snapshot_dir,
                                         'epoch-last.pth')
        try:
            shutil.copy(current_epoch_checkpoint, last_epoch_checkpoint)
        except Exception as e:
            logger.warning(f"Could not copy checkpoint to {last_epoch_checkpoint}: {e}")
            pass

    def restore_checkpoint(self):
        t_start = time.time()
        # Load to the primary device specified during Engine initialization
        # If self.device is CPU, map_location will be 'cpu'.
        # If self.device is cuda:X, map_location will be 'cuda:X'.
        tmp = torch.load(self.continue_state_object, map_location=self.device)
        t_ioend = time.time()

        # The model instance (self.state.model) should already be created.
        # If it's going to be DataParallel, it should be wrapped *before* restoring.
        # load_model expects the state_dict for the raw model (without 'module.' prefix)

        actual_model = self.state.model.module if isinstance(self.state.model,
                                                             torch.nn.DataParallel) else self.state.model

        # Ensure the actual_model's parameters are on the primary device before loading state
        # This helps if load_model or optimizers expect parameters to be on a certain device.
        actual_model.to(self.device)

        load_model(actual_model, tmp['model'], True)  # True for strict loading

        if 'optimizer_l' in tmp and self.state.optimizer_l is not None:
            self.state.optimizer_l.load_state_dict(tmp['optimizer_l'])
            # Move optimizer states to the primary device if necessary
            for state in self.state.optimizer_l.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        if 'optimizer_r' in tmp and self.state.optimizer_r is not None:
            self.state.optimizer_r.load_state_dict(tmp['optimizer_r'])
            for state in self.state.optimizer_r.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        if 'optimizer' in tmp and self.state.optimizer is not None:
            self.state.optimizer.load_state_dict(tmp['optimizer'])
            for state in self.state.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        self.state.epoch = tmp['epoch'] + 1
        self.state.iteration = tmp['iteration']
        del tmp
        t_end = time.time()
        logger.info(
            "Load checkpoint from file {}, "
            "Time usage:\n\tIO: {}, restore snapshot: {}".format(
                self.continue_state_object, t_ioend - t_start, t_end - t_ioend))

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        if self.device.type == 'cuda':  # Check if primary device is cuda
            torch.cuda.empty_cache()
        if type is not None:
            logger.warning(
                "An exception occurred during Engine execution: {} {}".format(type, value) +
                "give up running process")
            return False