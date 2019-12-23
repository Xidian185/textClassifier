# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

#用于生成测试集和训练集
class DataSet(object):
    def __init__(self, config):
        self.config = config
		self._dataSource = config.dataSource
		self._stopWordSource = config.stopWordSource
		
		self._sequenceLength = config.sequenceLength
		self._embeddingSize = config.embeddingSize
		self._batchSize = config.batchSize
		self._rate = config.rate
		
		self._stopWordDict = {}
		self.trainViews = []
		self.trainLabels = []
		
		self.evalViews = []
		self.evalLabels = []
		
		self.wordEmbedding = None
		self.indexFreqs = []  #统计每个单词出现多少个review中
		
		self.labelList = []
		
	def _readData(self, filePath):
		#从csv文件中读取数据集
		df = pd.read_csv(filePath)
		
		if self.config.numClasses == 1:
			labels = df["sentiment"].tolist()
		elif self.config.numClasses > 1:
			labels = df["rate"].tolist()
			
		review = df["review"].tolist()
		reviews = [line.strip().split() for line in review]  #双层链表
		
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
		rainLables = np.asarray(y[:trainIndex], dtype="float32")
		
		evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
		evalLabels = np.asarray(reviews[trainIndex:], dtype="float32")
		
		return trainReviews, trainLabels, evalReviews, evalLabels
		
		
		
		
		
		
		
		
			
			
			
			
			