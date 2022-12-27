#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import datetime
# public
import torch


class Base_Trainer(object):
    """docstring for Base_Trainer"""
    def __init__(self, config):
        super(Base_Trainer, self).__init__()
        self.config = config
        self.__update_config()
        self.__initialize()
        self.setup_log_dict()
        self.setup_results_dict()

    def __update_config(self):
        # global configurations always not changed
        self.config.shuffle = True  # to shuffle the training set
        self.config.pin_memory = True  # pin memory for data loader
        self.config.drop_last = True  # drop the last training batch

    def __initialize(self):
        # some default variables and placeholders
        self.step, self.epoch, self.val_epoch = 0, 0, 0
        self.done = False
        self.dataloader_dict = {}
        self.valid_epoch = 0
        self.config.eva_modes = []
        if self.config.val:
            self.config.eva_modes.append('val')
        if self.config.test:
            self.config.eva_modes.append('test')

    def setup_log_dict(self):
        self.log_dict = {}
        for mode in ['train'] + self.config.eva_modes:
            self.log_dict[mode] = {}
            self.log_dict[mode]['eval'] = []
            self.log_dict[mode]['best_eval'] = []
        self.log_dict['start_time'] = datetime.datetime.now()
        self.log_dict['best_val_metric'] = 0.

    def setup_results_dict(self):
        self.results_dict = {}
        for mode in ['train'] + self.config.eva_modes:
            self.results_dict[mode] = {}
            self.results_dict[mode]['results'] = {}
            self.results_dict[mode]['best_results'] = {}

    def save_ckpt(self):
        # define the checkpoin to be saved
        checkpoint_to_save = {
        'step': self.step
        , 'epoch': self.epoch
        , 'log_dict': self.log_dict
        , 'model': self.model.state_dict()
        , 'optimizer': self.optimizer.state_dict()
        , 'scheduler': self.scheduler.state_dict()
        }
        # save the check point
        torch.save(checkpoint_to_save, self.config.CKPT_PT)

    def load_ckpt(self):
        ckpt_to_load =  torch.load(self.config.CKPT_PT, map_location=self.config.device) 
        self.step = ckpt_to_load['step']
        self.epoch = ckpt_to_load['epoch']
        self.log_dict = ckpt_to_load['log_dict']
        self.model.load_state_dict(ckpt_to_load['model'])
        self.optimizer.load_state_dict(ckpt_to_load['optimizer'])
        self.scheduler.load_state_dict(ckpt_to_load['scheduler'])