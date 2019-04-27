# -*- coding: utf-8 -*-
# @Author: Max
# @Date:   2018-01-19 11:33:37
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-19 11:33:56


"""
Alphabet maps objects to integer ids. It provides two way mapping from the index to the objects.
"""
import json
import os


class Alphabet:
    def __init__(self, name, label=False, keep_growing=True):
        """
        初始化一个指定了名字的字母表
        :param name: 字母表的名字
        :param label:  是否使用默认字符</unk>来填充第一个位置
        :param keep_growing: 失配是否可以继续增长
        """
        self.__name = name
        self.UNKNOWN = "</unk>"
        self.label = label
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.default_index = 0
        self.next_index = 1
        if not self.label:
            self.add(self.UNKNOWN)

    def clear(self, keep_growing=True):
        """
        清除字母表里保存的内容，将下标计数复位
        :param keep_growing:
        :return:
        """
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.default_index = 0
        self.next_index = 1
        
    def add(self, instance):
        """
        为新加入的词添加到词表（列表），并记录其下标，保存词->下标的映射
        :param instance:
        :return:
        """
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        """
        返回输入实体在字母表中的下标（若存在）
        :param instance:
        :return:
        """
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:       #若创建时参数为 可增长的， 则把失配词插入
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.instance2index[self.UNKNOWN]    # 设置为不可增长，则返回一个Unknown

    def get_instance(self, index):
        """
        返回输入下标对应的实体（词/ label）, 注意unknown占了一个位置，但却不在表里
        :param index:
        :return:
        """
        if index == 0:  # 0号元素虽然用</unk>占位了，但是list里面并没有，返回None
            # First index is occupied by the wildcard element.
            return None
        try:
            return self.instances[index - 1]
        except IndexError:
            print('WARNING:Alphabet get_instance ,unknown instance, return the first label.')
            return self.instances[0]

    def size(self):
        """
        返回词表的长度大小（+1，因为有unknown）
        :return:
        """
        # if self.label:
        #     return len(self.instances)
        # else:
        return len(self.instances) + 1

    def iteritems(self):
        """
        返回一个 实体-》下标映射 结构的迭代器
        :return:
        """
        return self.instance2index.iteritems()

    def enumerate_items(self, start=1):
        """
        返回一个从指定下标start开始，到字母表结束的 下标->字母 映射结构
        :param start:
        :return:
        """
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is allowed between [1 : size of the alphabet)")
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        """
        设置字母表 不可以 继续插入
        :return:
        """
        self.keep_growing = False

    def open(self):
        """
        设置字母表 允许 继续插入
        :return:
        """
        self.keep_growing = True

    def get_content(self):
        """
        将字母表中保存的内容封装在 json数据结构
        :return:
        """
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:  ## 直接将保存为json结构后的数据 转化为json字符串文件保存
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        except Exception as e:
            print("Exception: Alphabet is not saved: %s" % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
