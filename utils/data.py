# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-14 17:34:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-29 15:26:51
import sys
import numpy as np
from utils.alphabet import Alphabet
from utils.functions import *
from utils.gazetteer import Gazetteer
import time


START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"

class Data:
    def __init__(self): 
        self.MAX_SENTENCE_LENGTH = 350
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True  #对文字中出现的数字进行归一化， 所有数字置零
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.norm_gaz_emb = False
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword')
        self.char_alphabet = Alphabet('character')
        # self.word_alphabet.add(START)
        # self.word_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(START)
        # self.char_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(PADDING)
        self.label_alphabet = Alphabet('label', True)
        self.gaz_lower = False
        self.gaz = Gazetteer(self.gaz_lower)
        self.gaz_alphabet = Alphabet('gaz')  # vec词典中，能被训练、dev和测试数据找到的词
        self.HP_fix_gaz_emb = False
        self.HP_use_gaz = True

        self.tagScheme = "NoSeg"
        self.char_features = "LSTM" 

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []
        self.use_bigram = True # train: False
        self.word_emb_dim = 50 # 解析文件时修改
        self.biword_emb_dim = 10 # 解析文件时修改 (就没有）
        self.char_emb_dim = 10 # 解析文件时修改 （实际里把word当作了char，只提了word）
        self.gaz_emb_dim = 50 # 解析文件时修改
        self.gaz_dropout = 0.5
        self.pretrain_word_embedding = None # 解析文件时修改
        self.pretrain_biword_embedding = None # 解析文件时修改
        self.pretrain_gaz_embedding = None # 解析文件时修改
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        ### hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 10 # train: 1
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200 # lstm隐藏层维度
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_use_char = False
        self.HP_gpu = False
        self.HP_lr = 0.075
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0

        
    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s"%(self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s"%(self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s"%(self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Use          bigram: %s"%(self.use_bigram))
        print("     Word  alphabet size: %s"%(self.word_alphabet_size))
        print("     Biword alphabet size: %s"%(self.biword_alphabet_size))
        print("     Char  alphabet size: %s"%(self.char_alphabet_size))
        print("     Gaz   alphabet size: %s"%(self.gaz_alphabet.size()))
        print("     Label alphabet size: %s"%(self.label_alphabet_size))
        print("     Word embedding size: %s"%(self.word_emb_dim))
        print("     Biword embedding size: %s"%(self.biword_emb_dim))
        print("     Char embedding size: %s"%(self.char_emb_dim))
        print("     Gaz embedding size: %s"%(self.gaz_emb_dim))
        print("     Norm     word   emb: %s"%(self.norm_word_emb))
        print("     Norm     biword emb: %s"%(self.norm_biword_emb))
        print("     Norm     gaz    emb: %s"%(self.norm_gaz_emb))
        print("     Norm   gaz  dropout: %s"%(self.gaz_dropout))
        print("     Train instance number: %s"%(len(self.train_texts)))
        print("     Dev   instance number: %s"%(len(self.dev_texts)))
        print("     Test  instance number: %s"%(len(self.test_texts)))
        print("     Raw   instance number: %s"%(len(self.raw_texts)))
        print("     Hyperpara  iteration: %s"%(self.HP_iteration))
        print("     Hyperpara  batch size: %s"%(self.HP_batch_size))
        print("     Hyperpara          lr: %s"%(self.HP_lr))
        print("     Hyperpara    lr_decay: %s"%(self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s"%(self.HP_clip))
        print("     Hyperpara    momentum: %s"%(self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s"%(self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s"%(self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s"%(self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s"%(self.HP_bilstm))
        print("     Hyperpara         GPU: %s"%(self.HP_gpu))
        print("     Hyperpara     use_gaz: %s"%(self.HP_use_gaz))
        print("     Hyperpara fix gaz emb: %s"%(self.HP_fix_gaz_emb))
        print("     Hyperpara    use_char: %s"%(self.HP_use_char))
        if self.HP_use_char:
            print("             Char_features: %s"%(self.char_features))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file,'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                label = pairs[-1]
                self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        print("Refresh label alphabet finished: old:%s -> new:%s"%(old_size, self.label_alphabet_size))

    def build_alphabet(self, input_file):
        """
        逐行解析输入（序列标注数据）文件，取出文本字符，及字符的label
        """
        build_start = time.time()
        in_lines = open(input_file, 'r').readlines()
        for idx in xrange(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 2:
                pairs = line.strip().split()     # 默认按空格分隔
                word = pairs[0].lower().decode('utf-8')  # 空格前面的是汉字
                if self.number_normalized:       #所有的数字都按0处理
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)   #
                self.word_alphabet.add(word)     # label和单词存入字母表

                #二元字符信息：当前字和紧跟的下一个字（若有）
                if self.use_bigram:
                    if idx < len(in_lines) - 1 and len(in_lines[idx+1]) > 2:
                        biword = word + in_lines[idx+1].strip().split()[0].lower().decode('utf-8')
                    else:
                        biword = word + NULLKEY
                    self.biword_alphabet.add(biword)
                if self.HP_use_char:
                    for char in word:  ##明明都只是字但还是按字符拆分？
                        self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.biword_alphabet_size = self.biword_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        ##
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        build_end = time.time()
        print("build alphabet for file %s cost time: %.2fs" % (input_file, build_end - build_start))


    def build_gaz_file(self, gaz_file):
        ## build gaz file,initial read gaz embedding file
        """
        读取文件并把word2vec的word存入self.gaz中。形式为(word, "one_source"
        :param gaz_file:
        :return:
        """
        build_start = time.time()
        if gaz_file:
            fins = open(gaz_file, 'r').readlines()
            for fin in fins:
                fin = fin.strip().split()[0].lower().decode('utf-8')
                if fin:
                    self.gaz.insert(fin, "one_source")
            print("Load gaz file: ", gaz_file, " total size:", self.gaz.size())
        else:
            print("Gaz file is None, load nothing")
        build_end = time.time()
        print("build alphabet for gaz file %s cost time: %.2fs" % (gaz_file, build_end - build_start))


    def build_gaz_alphabet(self, input_file):
        """
        对于每一段可以在vec中查找到的词序列，构建一个训练、测试、dev共用的索引字母表
        :param input_file:
        :return:
        """
        build_start = time.time()
        in_lines = open(input_file, 'r').readlines()
        word_list = []
        for line in in_lines:
            if len(line) > 2:  # 将一句话的所有词整合在一起
                word = line.split()[0].strip().lower().decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                word_list.append(word)   # 将连续的非O字符存在一起
            else:  # 一句话结束
                w_length = len(word_list)
                for idx in range(w_length):
                    # 用enumerateMatch来枚举查找当前串里的所有适配词（去尾
                    matched_entity = self.gaz.enumerateMatchList(word_list[idx:])  # 查找前枚举了起点，查找时枚举终点
                    for entity in matched_entity:
                        # print entity, self.gaz.searchId(entity),self.gaz.searchType(entity)
                        self.gaz_alphabet.add(entity)
                word_list = []
        print("gaz alphabet size:", self.gaz_alphabet.size())
        build_end = time.time()
        print("build alphabet for gaz file %s cost time: %.2fs" % (input_file, build_end - build_start))

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close() 
        self.gaz_alphabet.close()  

    def build_word_pretrain_emb(self, emb_path):
        print("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)

    def build_biword_pretrain_emb(self, emb_path):
        print("build biword pretrain emb...")
        self.pretrain_biword_embedding, self.biword_emb_dim = build_pretrain_embedding(emb_path, self.biword_alphabet, self.biword_emb_dim, self.norm_biword_emb)

    def build_gaz_pretrain_emb(self, emb_path):
        print("build gaz pretrain emb...")
        self.pretrain_gaz_embedding, self.gaz_emb_dim = build_pretrain_embedding(emb_path, self.gaz_alphabet,  self.gaz_emb_dim, self.norm_gaz_emb)

    """
    def generate_instance(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))
    """

    def generate_instance_with_gaz(self, input_file, name):
        """
        重新遍历输入文件，使用预定义的字母表，按每句话一个list嵌套list，获得每句话里的所有词、二元词、字、label、latiice word以及相应的字母表id
        :param input_file:
        :param name:
        :return:
        """
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet, self.biword_alphabet, self.use_bigram, self.char_alphabet, self.HP_use_char, self.gaz_alphabet,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz(input_file, self.gaz,self.word_alphabet, self.biword_alphabet, self.use_bigram, self.char_alphabet, self.HP_use_char, self.gaz_alphabet,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet, self.biword_alphabet, self.use_bigram, self.char_alphabet, self.HP_use_char, self.gaz_alphabet,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,self.biword_alphabet, self.use_bigram, self.char_alphabet, self.HP_use_char, self.gaz_alphabet,  self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))


    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file,'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
           content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert(sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')

            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s"%(name, output_file))





