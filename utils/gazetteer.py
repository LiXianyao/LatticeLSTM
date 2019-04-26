# -*- coding: utf-8 -*-
from trie import Trie

class Gazetteer:
    def __init__(self, lower):
        """
        词典包含的基本成分：
            一个用于查找的trie树
            一个词->来源的映射
            一个词->编号（递增）的映射（有了这些dict为什么还用字典树？）
            是否转小写的控制使能
            默认的分隔符
        :param lower:
        """
        self.trie = Trie()
        self.ent2type = {} ## word list to type
        self.ent2id = {"<UNK>": 0}   ## word list to id
        self.lower = lower
        self.space = ""

    def enumerateMatchList(self, word_list):
        """
        直接调用trie 中的枚举匹配，得到所有word_list中在词典里出现的前缀串
        :param word_list:
        :return:
        """
        if self.lower:
            word_list = [word.lower() for word in word_list]
        match_list = self.trie.enumerateMatch(word_list, self.space)
        return match_list

    def insert(self, word_list, source):
        if self.lower:  # 转小写
            word_list = [word.lower() for word in word_list]
        self.trie.insert(word_list)  # 插入trie树
        string = self.space.join(word_list)
        if string not in self.ent2type:  # 放入一个dict ent2type，记录每个word来自于哪里（这里指“one_source”）
            self.ent2type[string] = source
        if string not in self.ent2id:    # 放入一个dict ent2id，给每个word分配一个id
            self.ent2id[string] = len(self.ent2id)

    def searchId(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        string = self.space.join(word_list)
        if string in self.ent2id:
            return self.ent2id[string]
        return self.ent2id["<UNK>"]

    def searchType(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        string = self.space.join(word_list)
        if string in self.ent2type:
            return self.ent2type[string]
        print  "Error in finding entity type at gazetteer.py, exit program! String:", string
        exit(0)

    def size(self):
        return len(self.ent2type)




