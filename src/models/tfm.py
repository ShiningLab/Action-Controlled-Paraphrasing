#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# built-in

# public
import torch.nn as nn
from transformers import (
    AutoConfig
    , EncoderDecoderConfig
    , EncoderDecoderModel
    )
# private


class ModelGraph(nn.Module):
    """docstring for ModelGraph"""
    def __init__(self, config, **kwargs):
        super(ModelGraph, self).__init__()
        # initialize configurations
        self.config = config
        # update configurations
        self.update_config(**kwargs)
        # encoder
        encoder_config = AutoConfig.from_pretrained(self.config.ENCODER_PATH)
        # encoder_config.hidden_size = self.config.en_hidden_size
        # encoder_config.num_hidden_layers = self.config.en_num_hidden_layers
        # encoder_config.num_attention_heads = self.config.en_num_attention_heads
        # encoder_config.intermediate_size = self.config.en_intermediate_size
        # decoder
        decoder_config = AutoConfig.from_pretrained(self.config.DECODER_PATH)
        # decoder_config.hidden_size = self.config.de_hidden_size
        # decoder_config.num_hidden_layers = self.config.de_num_hidden_layers
        # decoder_config.num_attention_heads = self.config.de_num_attention_heads
        # decoder_config.intermediate_size = self.config.de_intermediate_size
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.decoder_start_token_id = self.config.bos_token_id
        # seq2seq
        tfm_config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=encoder_config
            , decoder_config=decoder_config
            )
        tfm_config.vocab_size = decoder_config.vocab_size
        tfm_config.bos_token_id = self.config.bos_token_id
        tfm_config.eos_token_id = self.config.eos_token_id
        tfm_config.pad_token_id = self.config.pad_token_id
        tfm_config.decoder_start_token_id = self.config.bos_token_id
        # initialize model
        self.model = EncoderDecoderModel(config=tfm_config)
        if self.config.pretrain == 'tfs':
            self.model.encoder.resize_token_embeddings(self.config.vocab_size)
            self.model.decoder.resize_token_embeddings(self.config.vocab_size)
        self.config.model_config = self.model.config

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def forward(self, xs, ys, zs=None):
        return self.model(
            **xs
            , labels=ys
            )

    def generate(self, xs):
        ys_ = self.model.generate(
            input_ids=xs.input_ids
            , max_new_tokens=self.config.max_new_tokens + 1  # eos
            , num_beams=self.config.num_beams
            , num_beam_groups=1
            , early_stopping=True
            , num_return_sequences=self.config.num_return_sequences
            , pad_token_id=self.config.pad_token_id
            , bos_token_id=self.config.bos_token_id
            , eos_token_id=self.config.eos_token_id
            , decoder_start_token_id=self.config.bos_token_id
        )
        return ys_