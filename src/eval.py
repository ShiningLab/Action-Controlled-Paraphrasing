#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import wandb
import evaluate
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore


class Evaluator(object):
    """docstring for Evaluator"""
    def __init__(self, results_dict, config):
        super(Evaluator, self).__init__()
        self.xs = results_dict['xs']
        self.ys = results_dict['ys']
        self.ys_ = results_dict['ys_']
        self.config = config
        self.init_metrics()
        self.get_metrics()
        self.get_info()

    def  init_metrics(self):
        # BLEU
        self.bleu = BLEUScore(n_gram=4)
        # iBLEU
        self.alphas = [0.7, 0.8, 0.9]
        self.sbleu = BLEUScore(n_gram=4)
        # ROUGEScore
        self.rouge = ROUGEScore(rouge_keys=('rouge1', 'rouge2', 'rougeL'))
        # others: METEOR
        self.others = evaluate.load(path=self.config.METRIC_PATH.format('meteor'))

    def get_metrics(self):
        # compute metrics
        bleu = self.bleu(self.ys_, [[y] for y in self.ys])
        sbleu = self.sbleu(self.ys_, [[x] for x in self.xs])
        ibleus = [alpha * bleu - (1 - alpha) * sbleu for alpha in self.alphas]
        rouge = {k:v.item() for k, v in self.rouge(self.ys_, self.ys).items()}
        others = self.others.compute(predictions=self.ys_, references=self.ys)
        # format metrics
        self.metrics_dict =  {
        'bleu': bleu
        , 'ibleu0.7': ibleus[0]
        , 'ibleu0.8': ibleus[1]
        , 'ibleu0.9': ibleus[2]
        , 'rouge1': rouge['rouge1_fmeasure']
        , 'rouge2': rouge['rouge2_fmeasure']
        , 'rougeL': rouge['rougeL_fmeasure']
        , 'meteor': others['meteor']
        }

    def get_info(self):
        # get info
        self.info = '|'
        for k, v in self.metrics_dict.items():
            self.info += '{}:{:.4f}|'.format(k, v*100)
        # update logger
        try:
            wandb.log(self.metrics_dict)
        except:
            pass