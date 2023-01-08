#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in

# public
import torch
# private
from config import Config
from src.utils import helper


class Paraphraser(object):
    """docstring for Paraphraser"""
    def __init__(self):
        super(Paraphraser, self).__init__()
        self.config = Config()
        self.update_config()
        self.initialize()

    def update_config(self):
        # setup device
        self.config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def initialize(self):
        # setup random seed
        helper.set_seed(self.config.seed)
        
    def train(self):
        # trainer
        self.trainer = helper.get_trainer(self.config)
        self.trainer.train()

    def test(self):
        # tester
        self.tester = helper.get_tester(self.trainer)
        self.tester.test()

def main():
    pp = Paraphraser()
    # train
    pp.train()
    # test
    pp.test()

if __name__ == '__main__':
      main()