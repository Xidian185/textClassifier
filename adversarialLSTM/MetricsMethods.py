# -*-coding:utf-8-*-

# 此文件定义全部的计算性能指标的函数

# 1、计算均值
def mean(item: list) -> float:
    return sum(item) / len(item) if len(item) > 0 else 0


# 2、计算准确率
def accuracy(pred_y, true_y):
    """
        计算二类和多类的准确率
        :param pred_y: 预测结果
        :param true_y: 真实结果
        :return:
    """
    # 预测值可能是两维列表
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    acc = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            acc += 1
    return acc / len(pred_y) if len(pred_y) > 0 else 0


# 3、真正率。预测为正的例子中确实为正的个数 / 预测为正的例子个数
def binary_precision(pred_y, true_y, positive=1):
    """
    二类的精确率计算
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    pos = 0  # 判断为正的
    tItem = 0  # 实际也为正的
    for i in range(len(pred_y)):
        if pred_y[i] == positive:
            pos += 1
            if true_y[i] == pred_y[i]:
                tItem += 1
    return tItem / pos if pos > 0 else 0


# 4、二分类召回率
def binary_recall(pred_y, true_y, positive=1):
    """
    :param pred_y: 预测结果
    :param true_y: 实际结果
    :param positive: 正例索引表示
    :return:
    """
    trueP = 0  # 实际正值
    preP = 0  # 实际为正，并且判断也为正
    for i in range(len(pred_y)):
        if true_y[i] == positive:
            trueP += 1
            if true_y[i] == pred_y[i]:
                preP += 1
    return preP / trueP if trueP > 0 else 0


# 5、二分类F值
def binary_f_beta(pred_y, true_y, beta=1.0, positive=1):
    """
    :param pred_y: 预测值
    :param true_y: 真实值
    :param beta: β值
    :param positive: 正例索引表示
    :return:
    """
    precision = binary_precision(pred_y, true_y)
    recall = binary_recall(pred_y, true_y)
    try:
        f_beta = (1 + beta + beta) * precision * recall / (beta * beta * precision) + recall
    except:
        f_beta = 0
    return f_beta

#6、多分类精确率。返回每一个分类中预测是i类的实例中确实为i的实例/实例i的个数
def multi_precision(pred_y, true_y, labels):
    """
    :param pred_y: 预测值
    :param true_y: 实际值
    :param labels: 各类实例索引表示
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [pred[0] for pred in pred_y]
    precision = [binary_precision(pred_y, true_y) for label in labels]
    return mean(precision)

#7、多分类召回率
def multi_recall(pred_y, true_y, labels):
    """
    多类的召回率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    recalls = [binary_recall(pred_y, true_y, label) for label in labels]
    rec = mean(recalls)
    return rec

#8、多分类F值
def multi_f_beta(pred_y, true_y, labels, beta=1.0):
    """
    多类的f beta值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :param beta: beta值
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    f_betas = [binary_f_beta(pred_y, true_y, beta, label) for label in labels]
    f_beta = mean(f_betas)
    return f_beta

#9、二分类各项指标
def get_binary_metrics(pred_y, true_y, f_beta=1.0):
    """
    得到二分类的性能指标
    :param pred_y:
    :param true_y:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = binary_recall(pred_y, true_y)
    precision = binary_precision(pred_y, true_y)
    f_beta = binary_f_beta(pred_y, true_y, f_beta)
    return acc, recall, precision, f_beta

#10、多分类多项指标
def get_multi_metrics(pred_y, true_y, labels, f_beta=1.0):
    """
    得到多分类的性能指标
    :param pred_y:
    :param true_y:
    :param labels:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = multi_recall(pred_y, true_y, labels)
    precision = multi_precision(pred_y, true_y, labels)
    f_beta = multi_f_beta(pred_y, true_y, labels, f_beta)
    return acc, recall, precision, f_beta