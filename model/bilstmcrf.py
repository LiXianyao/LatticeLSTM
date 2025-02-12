# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-05 23:15:17

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.bilstm import BiLSTM
from model.crf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, data):
        super(BiLSTM_CRF, self).__init__()
        print("build batched lstmcrf...")
        self.gpu = data.HP_gpu
        ## add two more label for downlayer lstm, use original label size for CRF？？？？
        label_size = data.label_alphabet_size  # 原值存临时
        data.label_alphabet_size += 2 ## ！！data中的被修改了
        self.lstm = BiLSTM(data)
        self.crf = CRF(label_size, self.gpu)


    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths,  char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        """ 调用模型，计算得到损失和标记序列
            内容与 forward基本一致，只不过多了一个调用crf层的损失计算及返回
        """
        outs = self.lstm.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths,  char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        return total_loss, tag_seq


    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        """ 只在eval时被使用，与neg_log_likelihood_loss的处理只差了一个损失计算 """
        outs = self.lstm.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        return tag_seq


    def get_lstm_features(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        return self.lstm.get_lstm_features(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        