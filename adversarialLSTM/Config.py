# -*- coding:utf-8 -*-

class TraingConfig(object):
    def __init__(self):
        self.epochs = 5  # 几次循环训练
        self.evaluateEvery = 100  # 多少次batch进行一次evaluate
        self.checkpointEvery = 100  # 多少次batch进行一次checkpoint保存
        self.learningRate = 0.001  # 优化函数学习率


class ModelConfig(object):
    def __init__(self):
        self.embeddingSize = 200
        self.hiddenSize = 128  # LSTM神经元个数
        self.dropoutKeepProb = 0.5
        self.l2RegLambda = 0.0  # 用户加入扰动
        self.epsilon = 5


class Config(object):
    def __init__(self):
        self.sequenceLength = 200  # 所有序列长度的均值
        self.batchSize = 128
        self.dataSource = "../data/preProcessed/labeledData.csv"
        self.stopWordSource = "../data/english"
        self.numClasses = 1  # 二分类设置为1，多分类设置为相应数字
        self.rate = 0.8  # 训练集比例
        self.training = TraingConfig()
        self.model = ModelConfig()

if __name__ == "__main__":
    config = Config()
    print(config.dataSource)