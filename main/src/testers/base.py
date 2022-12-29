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

    def __initialize(self):
        self.trainer.load_ckpt()
        self.step = self.trainer.step
        self.epoch = self.trainer.epoch
        self.val_epoch = self.trainer.val_epoch
        self.tokenizer = self.trainer.tokenizer
        self.model = self.trainer.model
        self.Dataset = self.trainer.Dataset