#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os, argparse
# public
from transformers import AutoConfig


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
    # lstm for seq2seq LSTM with Attention
    # tfm for vanilla transformer
    # em_tfm for transformer with pretrained embedding
    # en_tfm for transformer with pretrained encoder
    parser.add_argument('--model', type=str, default='lstm')
    # ori_quora for original Quora Question Pair
    # sep_quora for newly separated Quora Question Pair
    # twitterurl for original Twitter URL Paraphrasing
    parser.add_argument('--task', type=str, default='ori_quora')
    # mask control
    parser.add_argument('--mask', type=str2bool, default=False)
    # 0 for remove, 1 for keep, 2 for inference, 3 for padding
    parser.add_argument('--mask_size', type=int, default=4)
    parser.add_argument('--mask_delete_token_id', type=int, default=0)
    parser.add_argument('--mask_keep_token_id', type=int, default=1)
    parser.add_argument('--mask_infer_token_id', type=int, default=2)
    parser.add_argument('--mask_pad_token_id', type=int, default=3)
    # keep, random, copy, infer
    parser.add_argument(
        '--mask_weights', nargs='+', type=float, default=[0.2, 0.1, 0.1, 0.6])
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
    parser.add_argument('--bt', type=str2bool, default=False)  # back translate
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
    # model
    # encoder
    parser.add_argument('--en_hidden_size', type=int, default=450)
    parser.add_argument('--en_num_hidden_layers', type=int, default=3)
    parser.add_argument('--en_num_attention_heads', type=int, default=9)
    parser.add_argument('--en_intermediate_size', type=int, default=1024)
    # decoder
    parser.add_argument('--de_hidden_size', type=int, default=450)
    parser.add_argument('--de_num_hidden_layers', type=int, default=3)
    parser.add_argument('--de_num_attention_heads', type=int, default=9)
    parser.add_argument('--de_intermediate_size', type=int, default=1024)
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
    # mask control needs more epochs to fully converge
    parser.add_argument('--max_epoch', type=int, default=256)
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
        # mask weights
        if self.mask:
            self.mask_weights_str = '_'.join([str(_) for _ in self.mask_weights])
        else:
            self.mask_weights_str = str(False)
        # lstm
        if self.model == 'lstm':
            self.en_num_attention_heads = False
            self.en_intermediate_size = False
            self.de_num_attention_heads = False
            self.de_intermediate_size = False
            self.num_beams = False
        # en_tfm
        if self.model == 'en_tfm':
            self.encoder = 'bert_uncased_L-4_H-512_A-8'
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
        self.CKPT_PATH = os.path.join(
            self.RESOURCE_PATH, 'ckpts', self.task
            , str(self.mask), self.model, self.mask_weights_str, str(self.seed)
            )
        os.makedirs(self.CKPT_PATH, exist_ok=True)
        # log
        self.LOG_PATH = os.path.join(
            self.RESOURCE_PATH, 'log', self.task
            , str(self.mask), self.model, self.mask_weights_str, str(self.seed)
            )
        os.makedirs(self.LOG_PATH, exist_ok=True)
        # results
        self.RESULTS_PATH = os.path.join(
            self.RESOURCE_PATH, 'results', self.task
            , str(self.mask), self.model, self.mask_weights_str, str(self.seed)
            )
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        # test
        self.TEST_PATH = os.path.join(
            self.RESOURCE_PATH, 'test', self.task
            , str(self.mask),  self.model, self.mask_weights_str, str(self.seed)
            )
        os.makedirs(self.TEST_PATH, exist_ok=True)
        # update model hyper-parameters
        encoder_config = AutoConfig.from_pretrained(self.ENCODER_PATH)
        self.en_vocab_size = encoder_config.vocab_size
        decoder_config = AutoConfig.from_pretrained(self.DECODER_PATH)
        self.de_vocab_size = decoder_config.vocab_size
        match self.model:
            case 'em_tfm':
                self.en_hidden_size = encoder_config.hidden_size
                self.en_num_attention_heads = encoder_config.num_attention_heads
            case 'en_tfm':
                self.en_hidden_size = encoder_config.hidden_size
                self.en_num_hidden_layers = encoder_config.num_hidden_layers
                self.en_num_attention_heads = encoder_config.num_attention_heads
                self.en_intermediate_size = encoder_config.intermediate_size