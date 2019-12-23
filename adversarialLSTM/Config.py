# -*- coding:utf-8 -*-

class TraingConfig(object):
    def __init__(self):
        epochs = 5  #几次循环训练
		evaluateEvery = 100  #多少次batch进行一次evaluate
		checkpointEvery = 100  #多少次batch进行一次checkpoint保存
		learningRate = 0.001  #优化函数学习率
		
class ModelConfig(object):
	def __init__(self):
		embeddingSize = 200
		hiddenSize = 128  #LSTM神经元个数
		dropoutKeepProb = 0.5
		l2RegLambda = 0.0  #用户加入扰动
		epsilon = 5
		
class Config(self):
	def __init__(self):
		sequenceLength = 200  #所有序列长度的均值
		batchSize = 128
		dataSource = "../data/preProcessed/labeledData.csv"
		stopWordSource = "../data/english"
		numClasses = 1 #二分类设置为1，多分类设置为相应数字
		rate = 0.8 #训练集比例
		training = TraingConfig()
		model = ModelConfig()
		
config = Config()
		