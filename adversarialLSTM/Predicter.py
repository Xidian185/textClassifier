# -*-coding:utf-8 -*-
from Config import *
from DataSet import *
from Model import *

config = Config()

# 使用训练好的模型进行预测
x = "this movie is full of references like mad max ii the wild one and many others the ladybug´s face it´s a clear reference or tribute to " \
    "peter lorre this movie is a masterpiece we´ll talk much more about in the future"

with open("../data/wordJson/word2idx.json", "r", encoding="utf-8") as f:
    word2id = json.load(f)

with open("../data/wordJson/label2idx.json", "r", encoding="utf-8") as f:
    label2id = json.load(f)
# 制作label的id到标签的映射
id2label = {value: id for id, value in label2id.items()}

xIds = [word2id.get(word, word2id["UNK"]) for word in x.split(" ")]  # 不存在的单词的id使用UNK的
# 处理xIds的长度
if len(xIds) > config.sequenceLength:
    xIds = xIds[:config.sequenceLength]
elif len(xIds) < config.sequenceLength:
    xIds += [word2id["PAD"]] * (config.sequenceLength - len(xIds))

graph = tf.Graph()
with graph.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
    sess_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=sess_cfg)

    with sess.as_default():
        checkpoint_file = tf.train.latest_checkpoint("../model/adversarialLSTM/model/")  # 选择这个目录下最新的cpk
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))  # 根据meta创建saver
        saver.restore(sess, checkpoint_file)

        # 获得需要喂给模型的参数，输出的结果依赖的输入值
        inputX = graph.get_operation_by_name("inputX").outputs[0]
        dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]

        # 获得结果
        predictions = graph.get_tensor_by_name("loss/Bi-LSTM/predictions:0")

        #inputX: [xIds, xIds]是投机了，因为使用inputX: [xIds]会报维度错误，暂时没有解决
        result = sess.run(predictions, feed_dict={inputX: [xIds, xIds], dropoutKeepProb: 1.0})
        pred = result[0]

        print(pred)

        preLabel = [id2label[item] for item in pred]
        print(preLabel)
