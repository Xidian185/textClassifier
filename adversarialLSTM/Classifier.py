# -*-coding:utf-8 -*-
import tensorflow as tf
import datetime

from Config import *
from DataSet import *
from Model import *
from MetricsMethods import *

config = Config()
dataSet = DataSet(config)
dataSet.dataGen()

# 生成训练集和验证集
trainReviews = dataSet.trainReviews
trainLabels = dataSet.trainLabels
evalReviews = dataSet.evalReviews
evalLabels = dataSet.evalLabels

print("train data shape: {}".format(trainReviews.shape))
print("train label shape: {}".format(trainLabels.shape))
print("eval data shape: {}".format(evalReviews.shape))
print("eval label shape: {}".format(evalLabels.shape))

wordEmbedding = dataSet.wordEmbedding
indexFreqs = dataSet.indexFreqs
labelList = dataSet.labelList

with tf.Graph().as_default():
    # allow_soft_placement:是否允许指定运行设备；log_device_placement是否记录每一步在哪台设备运行
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率
    session = tf.Session(config=sess_config)

    with session.as_default():
        lstm = AdversarialBiLSTM(config, wordEmbedding, indexFreqs)
        # 定义一个全局跟踪计算步骤的变量
        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradAndVar = optimizer.compute_gradients(lstm.loss)
        ## 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradAndVar, global_step=globalStep)  # globalStep只会在train的环节中变化

        for grad, vars in gradAndVar:
            if grad is not None:
                tf.summary.histogram("{}/grad/hist".format(vars.name), grad)
                tf.summary.scalar("{}/grad/sparsity".format(vars.name),
                                  tf.nn.zero_fraction(grad))  # zero_fraction计算tensor中0的个数占比

        outDir = os.path.abspath(os.path.curdir + "summarys")
        lossSummary = tf.summary.scalar("loss", lstm.loss)
        summaryOp = tf.summary.merge_all()  # 到目前为止都是在定义summary操作

        # 定义train操作的summary写入器
        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, graph=session.graph)

        # 定义eval操作的summary写入器
        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, graph=session.graph)

        # tf.global_variables()赋值给var_list参数，定义Saver模型将保存的变量有哪些
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # 保存模型的一种方式，保存为pb文件
        savedModelPath = "../model/adversarialLSTM/savedModel"
        if os.path.exists(savedModelPath):
            os.rmdir(savedModelPath)
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)

        session.run(tf.initialize_all_variables())  # 将变量全部初始化


        # train一个batch将会进行的操作。返回损失值，准确率等参数。同时完成此batch的参数优化。
        def trainStep(batchX, batchY):
            feed_dict = {
                lstm.inputX: batchX,
                lstm.inputY: batchY,
                lstm.dropoutProb: config.model.dropoutKeepProb
            }
            _, summary, step, loss, predictions = session.run(
                [trainOp, summaryOp, globalStep, lstm.loss, lstm.predictions], feed_dict=feed_dict)
            if config.numClasses == 1:  # 二分类
                acc, recall, precision, f_beta = get_binary_metrics(predictions, batchY)
            elif config.numClasses > 1:
                acc, recall, precision, f_beta = get_multi_metrics(predictions, batchY, labelList)

            trainSummaryWriter.add_summary(summary, step)  # 认为summary是summaryOp运行后的句柄，可以关联所有summary操作

            return loss, acc, precision, recall, f_beta


        # eval一个batch将会进行的操作。返回损失值，准确率等参数。不进行参数优化，并且dropoutKeepProb设置为1.0
        def evalStep(batchX, batchY):
            feed_dict = {
                lstm.inputX: batchX,
                lstm.inputY: batchY,
                lstm.dropoutProb: 1.0
            }
            summary, step, loss, predictions = session.run([summaryOp, globalStep, lstm.loss, lstm.predictions],
                                                           feed_dict=feed_dict)
            if config.numClasses == 1:  # 二分类
                acc, recall, prec, f_beta = get_binary_metrics(predictions, batchY)
            elif config.numClasses > 1:
                acc, recall, prec, f_beta = get_multi_metrics(predictions, batchY, labelList)

            evalSummaryWriter.add_summary(summary, step)

            return loss, acc, recall, prec, f_beta


        # 进行epoches次循环的计算
        for i in range(config.training.epochs):
            for batchx, batchy in dataSet.nextBatch(trainReviews, trainLabels, config.batchSize):
                loss, acc, prec, recall, f_beta = trainStep(batchx, batchy)

                currentStep = tf.train.global_step(session, globalStep)
                # 打印这一步的训练情况
                print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                    currentStep, loss, acc, recall, prec, f_beta))

                if currentStep % config.training.evaluateEvery == 0:
                    # 每进行evaluateEvery次train,则进行一整批次的eval，统计整个eval集的模型计算结果
                    losses = []
                    accs = []
                    f_betas = []
                    precisions = []
                    recalls = []
                    for evalBatchx, evalBatchy in dataSet.nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc, recall, prec, f_beta = evalStep(evalBatchx, evalBatchy)
                        losses.append(loss)
                        accs.append(acc)
                        f_betas.append(f_beta)
                        precisions.append(prec)
                        recalls.append(recall)
                    timeC = datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(timeC,
                                                                                                         currentStep,
                                                                                                         mean(losses),
                                                                                                         mean(accs),
                                                                                                         mean(
                                                                                                             precisions),
                                                                                                         mean(recalls),
                                                                                                         mean(f_betas)))
                if currentStep % config.training.checkpointEvery == 0:
                    # 保存checkpoint文件
                    path = saver.save(session, "../model/adversarialLSTM/model/my-model", global_step=globalStep)
                    print("Saved model checkpoint to {}\n".format(path))

        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(lstm.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(lstm.dropoutProb)}
        outputs = {"predictions": tf.saved_model.utils.build_tensor_info(lstm.predictions)}

        sigunature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")  # tf.group将tensorflow组合

        # sigunature和legacy_init_op是builder的参数
        builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"prediction": sigunature},
                                             legacy_init_op=legacy_init_op)
        builder.save()

