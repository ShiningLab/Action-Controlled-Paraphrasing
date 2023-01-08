#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'



class Base_Tester(object):
    """docstring for Base_Tester"""
    def __init__(self, trainer):
        super(Base_Tester, self).__init__()
        self.trainer = trainer
        self.config = trainer.config
        self.__initialize()
        self.setup_test_dict()

    def __initialize(self):
        self.trainer.load_ckpt()
        self.step = self.trainer.step
        self.epoch = self.trainer.epoch
        self.val_epoch = self.trainer.val_epoch
        self.tokenizer = self.trainer.tokenizer
        self.model = self.trainer.model
        self.mask_modes = ['random', '0s', '1s', '2s', 'oracle']

    def setup_test_dict(self):
        self.test_dict = {}
        for mask_mode in self.mask_modes:
            self.test_dict[mask_mode] = {}
            self.test_dict[mask_mode] ['eval']= {}
            self.test_dict[mask_mode]['results'] = {}