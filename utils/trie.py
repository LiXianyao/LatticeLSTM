# -*- coding: utf-8 -*-
import collections
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)  # 通过这个defaultdict,使得可以只用一个for循环遍历构造trie
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        
        current = self.root
        for letter in word:  # 因为用了defaultdict，不需要递归，直接这样循环就可以建起来树了
            current = current.children[letter]
        current.is_word = True  # 修改节点标记，表示这个点到根的路径是一个词

    def search(self, word):
        """
        在字典树里查找输入词，不仅要适配，还要有词结束的标记
        :param word:
        :return:
        """
        current = self.root
        for letter in word:
            current = current.children.get(letter)

            if current is None:  # 因为default dict的关系，如果没有这条分支，取到的值就是none
                return False
        return current.is_word

    def startsWith(self, prefix):
        """
        在字典树里查找输入词(前缀)，只需要适配就可以了
        :param prefix:
        :return:
        """
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True

    def enumerateMatch(self, word, space="_", backward=False):
        """
        枚举查找输入序列在字符串里能找到的所有组合，每次枚举后舍去输入的最后一个字符
        即查找输入字符串的所有满足的前缀？？
        单字符的时候不处理（因为那是character based网络处理的内容）
        :param prefix:
        :return:
        """
        matched = []
        ## while len(word) > 1 does not keep character itself, while word keed character itself
        #"""
        if len(word) > 1:
            matched.extend(self.enum_search(word, space))
        """
        while len(word) > 1:
            if self.search(word):
                matched.append(space.join(word[:]))
            del word[-1]
        """
        return matched

    def enum_search(self, word, space="_"):
        """
        在字典树里查找输入词，不仅要适配，还要有词结束的标记
        :param word:
        :return:
        """
        current = self.root
        match_list = []
        matched_letter = []
        for letter in word:
            current = current.children.get(letter)

            if current is None:  # 因为default dict的关系，如果没有这条分支，取到的值就是none
                return match_list
            else:  ## 存在适配，记录适配路径上的字母
                matched_letter.append(letter)
                if current.is_word and len(matched_letter) > 1:
                    match_list.append(space.join(matched_letter[:]))
        return match_list
