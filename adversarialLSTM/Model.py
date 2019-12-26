# -*-coding:utf-8-*-

import tensorflow as tf
import numpy as np


# 定义模型，定义到出现损失值
class AdversarialBiLSTM(object):
    def __init__(self, config, wordEmbedding, wordFreqs):
        self.config = config
        # 定义模型输入
        self.inputX = tf.placeholder(tf.int64, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None], name="inputY")
        self.dropoutProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        # 根据单词出现的频率，处理单词向量
        weights = tf.reshape(tf.cast(wordFreqs / tf.reduce_sum(wordFreqs), tf.float32), [1, -1])

        # 处理词向量
        with tf.variable_scope("embedding"):
            # 利用词频计算新的词嵌入矩阵
            normWordEmbedding = self._normalize(tf.cast(wordEmbedding, tf.float32, name="word2vec"), weights)
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(normWordEmbedding, self.inputX)

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            with tf.variable_scope("Bi-LSTM", reuse=False):
                self.logits = self._Bi_LSTMAttention(self.embeddedWords)  # Bi_LSTMAttention计算的结果
                # 二分类和多分类计算损失值的方式不同
                if config.numClasses == 1:  # 二分类的话，logits中大于等于0则认为是1，否则认为是0
                    self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")
                    self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                          labels=tf.cast(
                                                                              tf.reshape(self.inputY, [-1, 1]),
                                                                              tf.float32))
                elif config.numClasses > 1:
                    self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")
                    self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
            loss = tf.reduce_mean(self.losses)

        with tf.name_scope("perturLoss"):  # 加入扰动
            with tf.variable_scope("Bi-LSTM", reuse=True):
                perturbEmbedded = self._addPerturbation(self.embeddedWords, loss)
                perturPredictions = self._Bi_LSTMAttention(perturbEmbedded)
                perturLosses = tf.nn.sigmoid_cross_entropy_with_logits(logits=perturPredictions,
                                                        labels=tf.cast(tf.reshape(self.inputY, [-1, 1]),
                                                                       dtype=tf.float32))
                perturLoss = tf.reduce_mean(perturLosses)

        self.loss = loss + perturLoss


    # 根据词出现的文档数，将词向量规范化
    def _normalize(self, embedding, weights):
        mean = tf.matmul(weights, embedding)
        powWordEmbedding = tf.pow(embedding - mean, 2)
        var = tf.matmul(weights, powWordEmbedding)  # 所有词向量的相应位置组成的序列的方差
        return (embedding - mean) / tf.sqrt(1e-6 + var)

    def _Bi_LSTMAttention(self, embeddedWords):
        config = self.config
        with tf.name_scope("Bi-LSTM"):
            # 前向lstm
            lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(num_units=config.model.hiddenSize, state_is_tuple=True),
                output_keep_prob=config.model.dropoutKeepProb)
            # 向后lstm
            lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(num_units=config.model.hiddenSize, state_is_tuple=True),
                output_keep_prob=config.model.dropoutKeepProb)
            outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, embeddedWords,
                                                                          dtype=tf.float32, scope="bi-lstm")
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]
            output = self._Attention(H)  # [batch_size, hidden_size]
            outputSize = config.model.hiddenSize

        with tf.name_scope("output"):
            outputW = tf.get_variable(
                name="outputW",
                shape=[outputSize, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses], name="outputB"))
            predictions = tf.matmul(output, outputW) + outputB
        return predictions

    def _Attention(self, H):
        """
       利用Attention机制得到句子的向量表示
       """
        hidden_size = self.config.model.hiddenSize  # 隐藏层个数
        W = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))  # 初始化一个向量
        M = tf.tanh(H)  # 激活函数处理成非线性

        # 对W和M做矩阵运算，M=[batch_size, sequenceLength, hidden_size]，计算前做维度转换成[batch_size * sequenceLength, hidden_size]
        # newM = [batch_size*sequenceLength, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hidden_size]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, sequenceLength]
        restoreM = tf.reshape(newM, [-1, self.config.sequenceLength])

        # 用softmax做归一化处理[batch_size, sequenceLength]
        softM = tf.nn.softmax(restoreM)

        # 利用求得的softM的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(softM, [-1, self.config.sequenceLength, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)

        rTanh = tf.tanh(sequeezeR)
        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(rTanh, self.dropoutProb)  # 模型运算时候的输入

        return output

    # 使用一般对抗  https://blog.csdn.net/qq_36489878/article/details/88575054
    def _addPerturbation(self, embedded, loss):
        """
        添加波动到word embedding
        """
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
        )
        grad = tf.stop_gradient(grad)
        perturb = self._scaleL2(grad, self.config.model.epsilon)
        return embedded + perturb

    def _scaleL2(self, x, norm_length):
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keepdims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keepdims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit
