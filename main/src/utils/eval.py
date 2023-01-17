#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# built-in
import copy, math
# public
import numpy as np
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('wordnet')
from torchmetrics.text.rouge import ROUGEScore

# adapted from https://github.com/guxd/C-DNPG/blob/94fec360bcc2cacfdf2dc85b6b74e75e406c69b0/learner.py#L40
class Metrics:
    def __init__(self):
        super(Metrics, self).__init__()
        # BLEU2, BLEU3, BLEU4
        self.bleu_weights = [(1./2., 1./2.), (1./3., 1./3., 1./3.), (1./4., 1./4., 1./4., 1./4.)]
        # BLEU smoothing function from Chen and Cherry (2014)
        self.chencherry = SmoothingFunction().method7
        self.rouge = ROUGEScore(
            tokenizer=word_tokenize
            , rouge_keys=('rouge1', 'rouge2', 'rougeL')
            )

    def sim_bleu(self, ref, hyp):
        """
        :param ref - tokenized reference, e.g., ['how', 'are', 'you']
        :param hyp - tokenized hypothesis, e.g., ['how', 'are', 'you']
        """
        return sentence_bleu(
            references=[ref]
            , hypothesis=hyp
            , weights=self.bleu_weights 
            , smoothing_function=self.chencherry
            )

    def sim_bleu4(self, ref, hyp):
        """
        :param ref - tokenized reference, e.g., ['how', 'are', 'you']
        :param hyp - tokenized hypothesis, e.g., ['how', 'are', 'you']
        """
        return sentence_bleu(
            references=[ref]
            , hypothesis=hyp
            , weights=(1./4., 1./4., 1./4., 1./4.)
            , smoothing_function=self.chencherry
            )

    def sim_meteor(self, ref, hyp):
        """
        :param refs - a list of tokenized references
        :param hyps - a list of tokenized hypothesis
        :return maxbleu - recall bleu
        :return avgbleu - precision bleu
        """
        return meteor_score([ref], hyp)

    def sim_rouge(self, ref, hyp):
        """
        :param ref - a list of  references
        :param hyp - a list of  hypothesis
        """
        rouge_scores = self.rouge(hyp, ref)
        rouge1_fmeasure = rouge_scores['rouge1_fmeasure'].item()
        rouge2_fmeasure = rouge_scores['rouge2_fmeasure'].item()
        rougeL_fmeasure = rouge_scores['rougeL_fmeasure'].item()
        return rouge1_fmeasure, rouge2_fmeasure, rougeL_fmeasure


class Evaluater(object):
    """docstring for Evaluater"""
    def __init__(self, xs, ys, ys_, config, loss, sample=True):
        super(Evaluater, self).__init__()
        self.Metrics = Metrics()
        self.config = config
        eval_size = self.config.eval_size if sample else -1
        self.ys = ys
        self.ys_ = ys_
        self.loss = loss
        self.tk_xs = [word_tokenize(x) for x in xs[:eval_size]]
        self.tk_ys = [word_tokenize(y) for y in ys[:eval_size]]
        self.tk_ys_ = [word_tokenize(y_) for y_ in ys_[:eval_size]]
        self.alphas = [0.9, 0.8, 0.7]
        self.get_eval()

    def get_eval(self):
        ibleu7_list, ibleu8_list, ibleu9_list = [], [], []
        bleu2_list, bleu3_list, bleu4_list= [], [], []
        meteor_list, rouge1_list, rouge2_list, rougeL_list = [], [], [], []
        for y, y_, tk_x, tk_y, tk_y_ in zip(self.ys, self.ys_, self.tk_xs, self.tk_ys, self.tk_ys_):
            bleu2, bleu3, bleu4 = self.get_bleu(tk_y, tk_y_)
            # alphas = [0.9, 0.8, 0.7]
            ibleu9, ibleu8, ibleu7 = self.get_ibleu(tk_x, tk_y_, bleu4)
            meteor = self.get_meteor(tk_y, tk_y_)
            rouge1, rouge2, rougeL = self.get_rouge_scores(y, y_)
            ibleu7_list.append(ibleu7)
            ibleu8_list.append(ibleu8)
            ibleu9_list.append(ibleu9)
            bleu2_list.append(bleu2)
            bleu3_list.append(bleu3)
            bleu4_list.append(bleu4)
            meteor_list.append(meteor)
            rouge1_list.append(rouge1)
            rouge2_list.append(rouge2)
            rougeL_list.append(rougeL)
        self.results = {}
        self.results['loss'] = self.loss
        self.results['perplexity'] = self.get_perplexity() if self.loss < float('inf') else float('inf')
        self.results['ibleu0.7'] = float(np.mean(ibleu7_list))
        self.results['ibleu0.8'] = float(np.mean(ibleu8_list))
        self.results['ibleu0.9'] = float(np.mean(ibleu9_list))
        self.results['bleu2'] = float(np.mean(bleu2_list))
        self.results['bleu3'] = float(np.mean(bleu3_list))
        self.results['bleu4'] = float(np.mean(bleu4_list))
        self.results['meteor'] = float(np.mean(meteor_list))
        self.results['rouge1'] = float(np.mean(rouge1_list))
        self.results['rouge2'] = float(np.mean(rouge2_list))
        self.results['rougeL'] = float(np.mean(rougeL_list))
        self.results['keymetric'] = self.results[self.config.keymetric]
        # evaluation info
        self.info = ''
        for k, v in self.results.items():
            self.info += ' {}:{:.4f}'.format(k, v)

    def get_perplexity(self):
        return math.exp(self.loss)

    def get_bleu(self, tk_y, tk_y_):
        return  self.Metrics.sim_bleu(tk_y, tk_y_) if tk_y_ else (0., 0., 0.)

    def get_ibleu(self, tk_x, tk_y_, bleu4):
        if bleu4:
            sbleu = self.Metrics.sim_bleu4(tk_x, tk_y_)
            ibleus = []
            for alpha in self.alphas:
                # https://aclanthology.org/P12-2008/
                ibleu = alpha * bleu4 - (1 - alpha) * sbleu
                ibleus.append(ibleu)
            return ibleus
        return 0., 0., 0.

    def get_meteor(self, tk_y, tk_y_):
        return self.Metrics.sim_meteor(tk_y, tk_y_) if tk_y_ else 0.

    def get_rouge_scores(self, y, y_):
        return self.Metrics.sim_rouge(y, y_) if y_ else (0., 0., 0.)