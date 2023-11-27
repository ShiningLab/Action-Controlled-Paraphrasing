#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os
# public
import torch
from lightning.pytorch import seed_everything
# private
from config import Config
from src import trainer


class ACP(object):
    """docstring for ACP"""
    def __init__(self):
        super(ACP).__init__()
        self.config = Config()
        self.initialize()

    def initialize(self):
        # setup device
        self.config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # get trainer
        self.trainer = trainer.S2STrainer(self.config)
        # setup random seed
        seed_everything(self.config.seed, workers=True)
        # enable tokenizer multi-processing
        if self.config.num_workers > 0:
            os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        # others
        torch.set_float32_matmul_precision('high')

def main() -> None:
    acp = ACP()
    acp.trainer.train()

if __name__ == '__main__':
      main()