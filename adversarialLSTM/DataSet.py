# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter
import gensim
import json

from Config import *


# 用于生成测试集和训练集
class DataSet(object):
    def __init__(self, cfg):
        self.config = cfg
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource

        self._sequenceLength = config.sequenceLength
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate

        self._stopWordDict = {}
        self.trainViews = []
        self.trainLabels = []

        self.evalViews = []
        self.evalLabels = []

        self.wordEmbedding = None
        self.indexFreqs = []  # 统计每个单词出现多少个review中

        self.labelList = []

    def _readData(self, filePath):
        # 从csv文件中读取数据集
        df = pd.read_csv(filePath)

        if self.config.numClasses == 1:
            labels = df["sentiment"].tolist()
        elif self.config.numClasses > 1:
            labels = df["rate"].tolist()

        review = df["review"].tolist()
        reviews = [line.strip().split() for line in review]  # 双层链表

        return reviews, labels

    def _labelToIndex(self, labels, label2index):
        """
        标签转换为索引
        """
        labelIds = [label2index[label] for label in labels]
        return labelIds

    def _wordToIndex(self, reviews, word2index):
        """
        将词转换为索引,如果单词不存在，则返回word2index["UNK"]
        """
        reviewIds = [[word2index.get(v, word2index["UNK"]) for v in review] for review in reviews]
        return reviewIds

    def _genTrainEvalData(self, x, y, word2ids, rate):
        """
        x和y是已经转换为id的数据
        """
        reviews = []
        for review in x:
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2ids["PAD"]] * (self._sequenceLength - len(review)))

        trainIndex = int(len(x) * rate)

        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.asarray(y[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.asarray(reviews[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels

    def _genVocabulary(self, reviews, labels):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """
        allwords = [v for review in reviews for v in review]

        # 去掉停词
        subWords = [word for word in allwords if word not in self._stopWordDict]

        # 去掉低频词
        counter = Counter(subWords)
        sortedCounter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        words = [item[0] for item in sortedCounter if item[1] > 5]
        voc, embedding = self._getWordEmbedding(words)
        self.wordEmbedding = embedding

        word2index = dict(zip(voc, range(len(voc))))
        self._getWordIndexFreq(voc, reviews, word2index)

        labelList = list(set(labels))  # 去重，然后转换为list
        label2index = dict(zip(labelList, range(len(labelList))))
        self.labelList = list(range(len(labelList)))  # 从0开始标记label

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("../data/wordJson/word2idx.json", "w", encoding="utf-8") as fw:
            json.dump(word2index, fw)

        with open("../data/wordJson/label2idx.json", "w", encoding="utf-8") as fl:
            json.dump(label2index, fl)

        return word2index, label2index

    # 词向量顺序和词汇表的顺序相同,返回词汇列表和相应的向量列表
    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """
        wordVec = gensim.models.KeyedVectors.load_word2vec_format("../data/word2Vec.bin", binary=True)
        voc = []
        embedding = []

        # 添加 "pad" 和 "UNK",
        voc.append("PAD")
        voc.append("UNK")
        embedding.append(np.zeros(self._embeddingSize))  # pad对应的向量
        embedding.append(np.random.randn(self._embeddingSize))  # UNK对应的向量

        for word in words:
            # 不在向量中的单词直接打印异常
            try:
                v = wordVec[word]
                voc.append(word)
                embedding.append(v)
            except:
                print(word + "不存在与词向量字典")
        return voc, np.asarray(embedding)

    # 计算每个单词出现的文章的个数
    def _getWordIndexFreq(self, vocab, reviews, word2idx):
        worddic = [dict(zip(review, range(len(review)))) for review in reviews]  # 一句话出现多个词的话，也只会统计一个
        freq = [0] * len(vocab)
        for word in vocab:
            count = 0
            for d in worddic:
                if word in d:
                    count += 1
            freq[word2idx[word]] = count
        self.indexFreqs = freq

    def _readStopWord(self, stopWordPath):
        """
        读取停用词
        """
        with open(stopWordPath, "r", encoding="utf-8") as f:
            file = f.read()
            lineList = file.splitlines()  # 以换行分割，一行内容为一个元素的列表
            stopWordDic = dict(zip(lineList, list(range(len(lineList)))))
            self._stopWordDict = stopWordDic

    def dataGen(self):
        """
        初始化训练集和验证集
        """
        # 读取原始数据
        reviews, labels = self._readData(self._dataSource)
        # 读取停顿次
        self._readStopWord(self._stopWordSource)

        word2index, label2index = self._genVocabulary(reviews, labels)
        reviewIds = self._wordToIndex(reviews, word2index)
        labelIds = self._labelToIndex(labels, label2index)

        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviewIds, labelIds, word2index,
                                                                                    self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

    def nextBatch(self, x, y, batchSize):
        total = len(x)
        times = total // batchSize
        for i in range(times):
            batchX = x[i * batchSize: (i + 1) * batchSize]
            batchY = y[i * batchSize: (i + 1) * batchSize]
            batchx = np.asarray(batchX, dtype="int64")
            batchy = np.asarray(batchY, dtype="float32")

            yield batchx, batchy


config = Config()
dataSet = DataSet(config)
dataSet.dataGen()
