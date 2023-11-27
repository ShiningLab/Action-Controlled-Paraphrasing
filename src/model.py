#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in

# public
import torch
import evaluate
import numpy as np
import lightning.pytorch as pl
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
# private
from src import helper


class S2S(pl.LightningModule):
    """docstring for S2S"""
    def __init__(self, config, tokenizer, **kwargs):
        super(S2S, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.update_config(**kwargs)
        self.get_metrics()
        self.model = helper.get_model(self.config)
        self.train_loss = []
        self.save_hyperparameters()

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def get_metrics(self):
        # BLEU
        self.val_bleu = BLEUScore(n_gram=4)
        # iBLEU
        self.alphas = [0.7, 0.8, 0.9]
        self.val_sbleu = BLEUScore(n_gram=4)
        # ROUGE
        self.val_rouge = ROUGEScore(rouge_keys=('rouge1', 'rouge2', 'rougeL'))
        # others: METEOR
        self.val_others = evaluate.load(path=self.config.METRIC_PATH.format('meteor'))

    def generate(self, xs):
        ys_ = self.model.generate(xs)
        ys_ = self.tokenizer.batch_decode(ys_, skip_special_tokens=True)
        return ys_

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        loss = self.model(xs, ys).loss
        self.train_loss.append(loss.item())
        self.log('train_step_loss', loss.item(), prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        loss = np.mean(self.train_loss, dtype='float32')
        self.log('train_epoch_loss', loss)
        self.train_loss = []

    def validation_step(self, batch, batch_idx):
        xs, raw_xs, raw_ys = batch
        ys_ = self.generate(xs)
        # update metrics
        self.val_bleu.update(ys_, [[y] for y in raw_ys])
        self.val_sbleu.update(ys_, [[x] for x in raw_xs])
        self.val_rouge.update(ys_, raw_ys)
        self.val_others.add_batch(predictions=ys_, references=raw_ys)

    def on_validation_epoch_end(self):
        val_bleu = self.val_bleu.compute().item()
        val_sbleu = self.val_sbleu.compute().item()
        val_ibleus = [alpha * val_bleu - (1 - alpha) * val_sbleu for alpha in self.alphas]
        val_rouge = {k:v.item() for k, v in self.val_rouge.compute().items()}
        val_others = self.val_others.compute()
        self.log_dict({
            'val_bleu': val_bleu
            , 'val_ibleu0.7': val_ibleus[0]
            , 'val_ibleu0.8': val_ibleus[1]
            , 'val_ibleu0.9': val_ibleus[2]
            , 'val_rouge1': val_rouge['rouge1_fmeasure']
            , 'val_rouge2': val_rouge['rouge2_fmeasure']
            , 'val_rougeL': val_rouge['rougeL_fmeasure']
            , 'val_meteor': val_others['meteor']
            })
        self.val_bleu.reset()
        self.val_sbleu.reset()
        self.val_rouge.reset()

    def predict_step(self, batch, batch_idx):
        xs, raw_xs, raw_ys = batch
        ys_ = self.generate(xs)
        return {
        'xs': raw_xs  # source text
        , 'ys': raw_ys  # target text
        , 'ys_': ys_  # output text
        }

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)]
                , "weight_decay": self.config.weight_decay
                }
            , {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)]
                , "weight_decay": 0.0
                }
            ]
        # optimizer = torch.optim.Adam(
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters
            , lr=self.config.learning_rate
            , eps=self.config.adam_epsilon
            )
        return optimizer