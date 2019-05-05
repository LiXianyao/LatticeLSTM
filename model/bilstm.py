# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-05-03 21:58:36
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from kblayer import GazLayer
from model.charbilstm import CharBiLSTM
from model.charcnn import CharCNN
from model.latticelstm import LatticeLSTM

class BiLSTM(nn.Module):
    def __init__(self, data):
        super(BiLSTM, self).__init__()
        print("build batched bilstm...")
        self.use_bigram = data.use_bigram # False
        self.gpu = data.HP_gpu
        self.use_char = data.HP_use_char # False
        self.use_gaz = data.HP_use_gaz
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0
        if self.use_char:
            """ 普通的CNN /LSTM 的 character-based network，项目中不用 """
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim
            if data.char_features == "CNN":
                self.char_feature = CharCNN(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_features == "LSTM":
                self.char_feature = CharBiLSTM(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            else:
                print("Error char feature selection, please check parameter data.char_features (either CNN or LSTM).")
                exit(0)
        self.embedding_dim = data.word_emb_dim # 获取词嵌入的长度（实际是字嵌入）
        self.hidden_dim = data.HP_hidden_dim # hidden = 1 * 200

        """ 设置（两个）dropout层和embedding层 """
        self.drop = nn.Dropout(data.HP_dropout) # embedding层的drop out
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.word_embeddings = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        self.biword_embeddings = nn.Embedding(data.biword_alphabet.size(), data.biword_emb_dim)

        self.bilstm_flag = data.HP_bilstm
        # self.bilstm_flag = False
        self.lstm_layer = data.HP_lstm_layer  # 几层lstm

        """ 加载word embedding层的初始数值（若有），否则直接做一个均匀分布采样的初始化 """
        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))
            
        if data.pretrain_biword_embedding is not None:
            self.biword_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_biword_embedding))
        else:
            self.biword_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.biword_alphabet.size(), data.biword_emb_dim)))
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        
        if self.bilstm_flag: # 如果是双向lstm，则单个lstm的hidden state长度减半
            lstm_hidden = data.HP_hidden_dim // 2  # //是结果向下取整的意思
        else:
            lstm_hidden = data.HP_hidden_dim
        lstm_input = self.embedding_dim + self.char_hidden_dim # 后一项只在单独做Character-Based时候不为0
        if self.use_bigram:
            lstm_input += data.biword_emb_dim

        """ 子层： LatticeLSTM，输入为 字embedding、lstm embedding； 分正向、反向 """
        self.forward_lstm = LatticeLSTM(lstm_input, lstm_hidden, data.gaz_dropout, data.gaz_alphabet.size(), data.gaz_emb_dim, data.pretrain_gaz_embedding, True, data.HP_fix_gaz_emb, self.gpu)
        if self.bilstm_flag:
            self.backward_lstm = LatticeLSTM(lstm_input, lstm_hidden, data.gaz_dropout, data.gaz_alphabet.size(), data.gaz_emb_dim, data.pretrain_gaz_embedding, False, data.HP_fix_gaz_emb, self.gpu)
        # self.lstm = nn.LSTM(lstm_input, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)

        """ 设置gpu使能（可以看到总共7个层件） """
        if self.gpu:
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.biword_embeddings = self.biword_embeddings.cuda()
            self.forward_lstm = self.forward_lstm.cuda()
            if self.bilstm_flag:
                self.backward_lstm = self.backward_lstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()


    def random_embedding(self, vocab_size, embedding_dim):
        """ 以均匀分布随机初始化一个权重矩阵作为embedding """
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def get_lstm_features(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
            ·获取、调用embedding层和Lattice-lstm层，获得lstm运算的结果，返回（接线性层）
            ·embedding层及Lattice层后均使用了一个dropout层
            input:
                gaz_list:  （batch, sent_len, variable 2 or 0） 【ids】
                word_inputs: (batch_size, sent_len) 【ids】
                bi_word_inputs: (batch_size, sent_len)【id】
                word_seq_lengths: list of batch_size, (batch_size,1 )
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output: 
                Variable(sent_len, batch_size, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        """ 设置embedding层，并进行dropout """
        word_embs =  self.word_embeddings(word_inputs) # 取得输入词（字）的所有embedding
        if self.use_bigram: # False， 不用
            biword_embs = self.biword_embeddings(biword_inputs)
            word_embs = torch.cat([word_embs, biword_embs],2) # 二元词embedding直接拼在输入embedding后面使用
        if self.use_char: # False
            ## calculate char lstm last hidden
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size,sent_len,-1)
            ## concat word and char together
            word_embs = torch.cat([word_embs, char_features], 2)
        word_embs = self.drop(word_embs) # 训练embedding层，随机置零一些元素，并把结果 * 1/(1-p) = 2
        # packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)

        """ 训练前向Lattice网络(self.forward_lstm)，获得输出及隐藏层状态 """
        hidden = None
        lstm_out, hidden = self.forward_lstm(word_embs, gaz_list, hidden)
        if self.bilstm_flag: # 是双向，计算反向lstm网络，并把正反向的结果拼接
            backward_hidden = None 
            backward_lstm_out, backward_hidden = self.backward_lstm(word_embs, gaz_list, backward_hidden)
            lstm_out = torch.cat([lstm_out, backward_lstm_out],2)
        # lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.droplstm(lstm_out) # 对双向lstm的结果进行dropout,返回输出（通向线性层）
        return lstm_out



    def get_output_score(self, gaz_list,  word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """ 获取 embedding -> (hidden ->) tag_seq 的计算结果矩阵，返回（交付CRF） """
        lstm_out = self.get_lstm_features(gaz_list, word_inputs,biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ## lstm_out (batch_size, sent_len, hidden_dim)
        outputs = self.hidden2tag(lstm_out)
        return outputs
    

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        ## mask is not used
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len
        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        outs = self.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        # outs (batch_size, seq_len, label_vocab)
        outs = outs.view(total_word, -1)
        score = F.log_softmax(outs, 1)
        loss = loss_function(score, batch_label.view(total_word))
        _, tag_seq  = torch.max(score, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        return loss, tag_seq


    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths,  char_inputs, char_seq_lengths, char_seq_recover, mask):
        """  只在eval时候使用 """
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len
        outs = self.get_output_score(gaz_list,  word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        outs = outs.view(total_word, -1)
        _, tag_seq  = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        ## filter padded position with zero
        decode_seq = mask.long() * tag_seq
        return decode_seq




