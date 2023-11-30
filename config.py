#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os, argparse
# private
from src import helper


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # paraphrase generation: qqp
    parser.add_argument('--data', type=str, default='qqp')
    # methods
    # base
    parser.add_argument('--method', type=str, default='base', choices=['base'])
    # model
    # pretrain
    # tfs: train from Scratch
    # en: encoder only
    # full: fully pretrained
    parser.add_argument('--pretrain', type=str, default='scratch', choices=['tfs', 'en', 'full'])
    # tfm for vanilla transformer
    parser.add_argument('--model', type=str, default='tfm', choices=['tfm'])
    # backbone
    # bert-base-uncased, bert_uncased_L-4_H-512_A-8
    parser.add_argument('--lm', type=str, default='bert_uncased_L-4_H-512_A-8')
    # encoder
    # parser.add_argument('--encoder', type=str, default='bert_uncased_L-4_H-512_A-8')
    # parser.add_argument('--en_hidden_size', type=int, default=512)
    # parser.add_argument('--en_num_hidden_layers', type=int, default=4)
    # parser.add_argument('--en_num_attention_heads', type=int, default=8)
    # parser.add_argument('--en_intermediate_size', type=int, default=2048)
    # decoder
    # parser.add_argument('--decoder', type=str, default='bert_uncased_L-4_H-512_A-8')
    # parser.add_argument('--de_hidden_size', type=int, default=512)
    # parser.add_argument('--de_num_hidden_layers', type=int, default=4)
    # parser.add_argument('--de_num_attention_heads', type=int, default=8)
    # parser.add_argument('--de_intermediate_size', type=int, default=2048)
    # training
    parser.add_argument('--load_ckpt', type=helper.str2bool, default=False)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=-1)  # -1 to enable infinite training
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--fast_dev_run', type=helper.str2bool, default=False)  # True for development
    # evaluation
    parser.add_argument('--monitor', type=str, default='val_ibleu0.8', choices=['val_ibleu0.8'])
    parser.add_argument('--patience', type=int, default=16)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--num_sanity_val_steps', type=int, default=2)
    parser.add_argument('--max_new_tokens', type=int, default=20)
    parser.add_argument('--num_beams', type=int, default=8)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    # trainer
    # 16-mixed, 32-true
    parser.add_argument('--precision', type=str, default='32-true')
    # (str, optional) Can be 'simple' or 'advanced'. Defaults to ''.
    parser.add_argument('--profiler', type=str, default='')
    # logger
    parser.add_argument('--offline', type=helper.str2bool, default=False)  # True for development
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--log_model', type=helper.str2bool, default=False)
    # save as argparse space
    return parser.parse_known_args()[0]


class Config(object):
    """docstring for Config"""
    def __init__(self):
        super(Config, self).__init__()
        self.update_config(**vars(init_args()))

    def update_config(self, **kwargs):
        # load config from parser
        for k,v in kwargs.items():
            setattr(self, k, v)
        # I/O
        self.CURR_PATH = './'
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        os.makedirs(self.DATA_PATH, exist_ok=True)
        self.DATA_PKL = os.path.join(self.DATA_PATH, f'{self.data}.pkl')
        # language model
        self.LM_PATH = os.path.join(self.RESOURCE_PATH, 'lm', self.lm)  # backbone
        self.ENCODER_PATH = self.LM_PATH
        self.DECODER_PATH = self.LM_PATH
        # model tokenizer
        self.TOKENIZER_PATH = os.path.join(self.RESOURCE_PATH, 'tokenizers', self.data, self.lm)
        # evaluation metrics
        self.METRIC_PATH = os.path.join(self.RESOURCE_PATH, 'metrics', '{}')
        # checkpoints
        self.CKPT_PATH = os.path.join(
            self.RESOURCE_PATH, 'ckpts', self.data, self.pretrain, self.lm, self.model, self.method, str(self.seed)
            )
        os.makedirs(self.CKPT_PATH, exist_ok=True)
        self.CKPT_LAST = os.path.join(self.CKPT_PATH, 'last.ckpt')
        # log
        self.ENTITY = 'mrshininnnnn'
        self.PROJECT = 'ACP'
        self.NAME = f'{self.data}-{self.pretrain}-{self.lm}-{self.model}-{self.method}-{self.seed}'
        self.LOG_PATH = os.path.join(
            self.RESOURCE_PATH, 'log', self.data, self.pretrain, self.lm, self.model, self.method
            )
        os.makedirs(self.LOG_PATH, exist_ok=True)
        self.LOG_TXT = os.path.join(self.LOG_PATH, f'{self.seed}.txt')
        os.remove(self.LOG_TXT) if os.path.exists(self.LOG_TXT) else None
        # results
        self.RESULTS_PATH = os.path.join(
            self.RESOURCE_PATH, 'results', self.data, self.pretrain, self.lm, self.method, self.model
            )
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        self.RESULTS_PKL = os.path.join(self.RESULTS_PATH, f'{self.seed}.pkl')