#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os, argparse
# public
from transformers import EncoderDecoderConfig


# helper function
def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def init_args():
    parser = argparse.ArgumentParser()
    # random seed
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load_ckpt', type=str2bool, default=False)
    # ori_quora for original Quora Question Pair
    # sep_quora for newly separated Quora Question Pair
    parser.add_argument('--task', type=str, default='sep_quora')
    # mask control
    parser.add_argument('--mask', type=str2bool, default=True)
    # 0 for remove, 1 for keep, 2 for inference, 3 for padding
    parser.add_argument('--mask_size', type=int, default=4)
    parser.add_argument('--mask_delete_token_id', type=int, default=0)
    parser.add_argument('--mask_keep_token_id', type=int, default=1)
    parser.add_argument('--mask_infer_token_id', type=int, default=2)
    parser.add_argument('--mask_pad_token_id', type=int, default=3)
    parser.add_argument('--mask_weights', type=list, default=[0.8, 0.1, 0.1])
    # data augmentation
    parser.add_argument(
        '--augs'
        , type=list
        , default=['x_x_copy', 'y_x_switch', 'ld', 'lc', 'bt']
        )
    parser.add_argument('--x_x_copy', type=str2bool, default=False)
    parser.add_argument('--y_x_switch', type=str2bool, default=False)
    parser.add_argument('--ld', type=str2bool, default=False)  # linear decompose
    parser.add_argument('--lc', type=str2bool, default=False)  # linear compose
    parser.add_argument('--bt', type=str2bool, default=True)  # back translate
    parser.add_argument('--lc_low', type=int, default=2)
    parser.add_argument('--lc_compo_size', type=int, default=8)
    parser.add_argument('--bt_src_lang', type=str, default='en') 
    parser.add_argument('--bt_tgt_lang', type=str, default='fr') 
    # pretrained language model
    parser.add_argument('--encoder', type=str, default='bert-base-uncased')
    parser.add_argument('--decoder', type=str, default='bert-base-uncased')
    parser.add_argument('--scorer', type=str, default='deberta-large-mnli')
    parser.add_argument('--src_translator', type=str, default='opus-mt-ROMANCE-en')
    parser.add_argument('--tgt_translator', type=str, default='opus-mt-en-ROMANCE')
    # tfm for vanilla transformer
    parser.add_argument('--model', type=str, default='tfm')
    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--num_attention_heads', type=int, default=9)
    parser.add_argument('--intermediate_size', type=int, default=1024)
    # data
    parser.add_argument('--stemming', type=str2bool, default=True)
    parser.add_argument('--en_max_len', type=int, default=20)
    parser.add_argument('--de_max_len', type=int, default=20)
    # training
    parser.add_argument('--val', type=str2bool, default=True)
    parser.add_argument('--test', type=str2bool, default=True)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    # evaluation
    parser.add_argument('--keymetric', type=str, default='ibleu0.8')  # validation
    parser.add_argument('--val_patience', type=int, default=32)  # w.r.t. epoch
    parser.add_argument('--eval_size', type=int, default=4000)  # for a fast training evaluation
    parser.add_argument('--num_beams', type=int, default=8)
    # save as argparse space
    return parser.parse_known_args()[0]


class Config():
    # config settings
    def __init__(self):
        super(Config, self).__init__()
        self.update_config(**vars(init_args()))
        
    def update_config(self, **kwargs):
        # load config from parser
        for k,v in kwargs.items():
            setattr(self, k, v)
        # I/O
        self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        # data
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        os.makedirs(self.DATA_PATH, exist_ok=True)
        self.DATA_PKL = os.path.join(self.DATA_PATH, f'{self.task}.pkl')
        # language model
        self.LM_PATH = os.path.join(self.RESOURCE_PATH, 'lm')
        self.ENCODER_PATH = os.path.join(self.LM_PATH, self.encoder)
        os.makedirs(self.ENCODER_PATH, exist_ok=True)
        self.DECODER_PATH = os.path.join(self.LM_PATH, self.decoder)
        os.makedirs(self.DECODER_PATH, exist_ok=True)
        self.SCORER_PATH = os.path.join(self.LM_PATH, self.scorer)
        os.makedirs(self.SCORER_PATH, exist_ok=True)
        self.SOURCE_TRANSLATOR_PATH = os.path.join(self.LM_PATH, self.src_translator)
        os.makedirs(self.SOURCE_TRANSLATOR_PATH, exist_ok=True)
        self.TARGET_TRANSLATOR_PATH = os.path.join(self.LM_PATH, self.tgt_translator)
        os.makedirs(self.TARGET_TRANSLATOR_PATH, exist_ok=True)
        # checkpoints
        self.train_size, self.val_size, self.test_size = 310302, 4000, 20000
        self.CKPT_PATH = os.path.join(
            self.RESOURCE_PATH, 'ckpts', str(self.mask), self.task, self.model, str(self.seed)
            )
        os.makedirs(self.CKPT_PATH, exist_ok=True)
        self.CKPT_PT = f'{self.train_size}_{self.val_size}_{self.test_size}.pt'
        self.CKPT_PT = os.path.join(self.CKPT_PATH, self.CKPT_PT)
        # log
        self.LOG_PATH = os.path.join(
            self.RESOURCE_PATH, 'log', str(self.mask), self.task, self.model, str(self.seed)
            )
        os.makedirs(self.LOG_PATH, exist_ok=True)