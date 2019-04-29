# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-05-12 22:09:37
import sys
import numpy as np
from alphabet import Alphabet
NULLKEY = "-null-"
def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet, char_alphabet, label_alphabet, number_normalized,max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    in_lines = open(input_file,'r').readlines()
    instance_texts = []
    instance_Ids = []
    words = []
    chars = []
    labels = []
    word_Ids = []
    char_Ids = []
    label_Ids = []
    for line in in_lines:
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0].decode('utf-8')
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if (max_sent_length < 0) or (len(words) < max_sent_length):
                instance_texts.append([words, chars, labels])
                instance_Ids.append([word_Ids, char_Ids,label_Ids])
            words = []
            chars = []
            labels = []
            word_Ids = []
            char_Ids = []
            label_Ids = []
    return instance_texts, instance_Ids


def read_seg_instance(input_file, word_alphabet, biword_alphabet, char_alphabet, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    in_lines = open(input_file,'r').readlines()
    instance_texts = []
    instance_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    for idx in xrange(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0].decode('utf-8')
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            words.append(word)
            if idx < len(in_lines) -1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0].decode('utf-8')
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_Ids.append(biword_alphabet.get_index(biword))
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if (max_sent_length < 0) or (len(words) < max_sent_length):
                instance_texts.append([words, biwords, chars, labels])
                instance_Ids.append([word_Ids, biword_Ids, char_Ids,label_Ids])
            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []
    return instance_texts, instance_Ids


def read_instance_with_gaz(input_file, gaz, word_alphabet, biword_alphabet, char_alphabet, gaz_alphabet, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    in_lines = open(input_file,'r').readlines()
    instance_texts = []
    instance_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    for idx in xrange(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0].decode('utf-8')
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            if idx < len(in_lines) -1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0].decode('utf-8')
            else:
                biword = word + NULLKEY
            """
            重新遍历训练、dev和测试数据文件，将其中的词转化为id映射，并保存在相应的结构
            """
            biwords.append(biword)
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_Ids.append(biword_alphabet.get_index(biword))
            label_Ids.append(label_alphabet.get_index(label))

            #词分字符，构造字符列表+id映射
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)

        else:  ##输入文件中， 句子与句子之间有一个空行
            ## 句子长度在最长句子长度范围限制内 （超过的就不用了，因为不好截断【会影响标注】）
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words)>0):
                """
                再次枚举所有可能的词，并获取它们的索引
                """
                gazs = []
                gaz_Ids = []
                w_length = len(words)
                # print sentence 
                # for w in words:
                #     print w," ",
                # print
                for idx in range(w_length):
                    ## 这里使用了这句话里的所有词，去头去尾枚举。也就是说包括了 O
                    #但是取id的时候可能会取到unknown
                    #也就是每次循环，会得到所有以当前字/词为开头的所有可能词 + 其id （失配的时候变成0）
                    matched_list = gaz.enumerateMatchList(words[idx:])
                    matched_length = [len(a) for a in matched_list]
                    # print idx,"----------"
                    # print "forward...feed:","".join(words[idx:])
                    # for a in matched_list:
                    #     print a,len(a)," ",
                    # print

                    # print matched_length

                    gazs.append(matched_list)
                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]
                    if matched_Id:
                        gaz_Ids.append([matched_Id, matched_length])
                    else:  ## matched_list 最坏情况下是 []， 相应的,id也这么赋值
                        gaz_Ids.append([])
                    
                instance_texts.append([words, biwords, chars, gazs, labels])
                instance_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids])

            ### 一句话处理完毕，它的所有词、二元词、字符等数据清空
            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []
            gazs = []
            gaz_Ids = []
    return instance_texts, instance_Ids


def read_instance_with_gaz_in_sentence(input_file, gaz, word_alphabet, biword_alphabet, char_alphabet, gaz_alphabet, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    in_lines = open(input_file,'r').readlines()
    instance_texts = []
    instance_Ids = []
    for idx in xrange(len(in_lines)):
        pair = in_lines[idx].strip().decode('utf-8').split()
        orig_words = list(pair[0])
        
        if (max_sent_length > 0) and (len(orig_words) > max_sent_length):
            continue
        biwords = []
        biword_Ids = []
        if number_normalized:
            words = []
            for word in orig_words:
                word = normalize_word(word)
                words.append(word)
        else:
            words = orig_words
        word_num = len(words)
        for idy in range(word_num):
            if idy < word_num - 1:
                biword = words[idy]+words[idy+1]
            else:
                biword = words[idy]+NULLKEY
            biwords.append(biword)
            biword_Ids.append(biword_alphabet.get_index(biword))
        word_Ids = [word_alphabet.get_index(word) for word in words]
        label = pair[-1]
        label_Id =  label_alphabet.get_index(label)
        gazs = []
        gaz_Ids = []
        word_num = len(words)
        chars = [[word] for word in words]
        char_Ids = [[char_alphabet.get_index(word)] for word in words]
        ## print sentence 
        # for w in words:
        #     print w," ",
        # print
        for idx in range(word_num):
            matched_list = gaz.enumerateMatchList(words[idx:])
            matched_length = [len(a) for a in matched_list]
            # print idx,"----------"
            # print "forward...feed:","".join(words[idx:])
            # for a in matched_list:
            #     print a,len(a)," ",
            # print
            # print matched_length
            gazs.append(matched_list)
            matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]
            if matched_Id:
                gaz_Ids.append([matched_Id, matched_length])
            else:
                gaz_Ids.append([])
        instance_texts.append([words, biwords, chars, gazs, label])
        instance_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Id])
    return instance_texts, instance_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):    
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim


       
def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    """  读取给定的embedding 文件 """
    embedd_dim = -1  # 不限制embedding的长度，根据读到的文件绑定
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:  # 检查embedding长度是否满足设置，若不等则抛出异常
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])  # embedding是一个 1*dim的行向量，np.empty()创建出来的是随机数矩阵
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0].decode('utf-8')] = embedd
    return embedd_dict, embedd_dim

if __name__ == '__main__':
    a = np.arange(9.0)
    print a
    print norm2one(a)
