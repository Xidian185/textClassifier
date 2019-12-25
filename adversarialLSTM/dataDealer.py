# -*- coding:utf-8 -*-
import pandas as pd
from bs4 import BeautifulSoup

#目标：1、去掉带标签和不带标签两个文件内容中的标点符号，2、带标签的文件取出rate列，3、拿到句子列表，后面用于生成词向量

with open(r"..\data\rawData\labeledTrainData.tsv", encoding="utf-8") as labeledF:
    labedL = [line.strip().split("\t") for line in labeledF.readlines() if len(line.strip().split("\t")) == 3]
    labedP = pd.DataFrame(labedL[1:], columns=labedL[0])

with open(r"..\data\rawData\unlabeledTrainData.tsv", encoding="utf-8") as unlabeledF:
    unlabedL = [line.strip().split("\t") for line in unlabeledF.readlines() if len(line.strip().split("\t")) == 2]
    unlabedP = pd.DataFrame(unlabedL[1:], columns=unlabedL[0])

def getRate(rateS):
    ss = rateS[1:-1].split("_")
    return ss[1]

labedP["rate"] = labedP["id"].apply(getRate)

#print(labedP.head())

#去掉特殊字符,同时将单词全部变成小写
def removeSen(sSen):
    obj = BeautifulSoup(sSen)
    text = obj.get_text()
    text = text.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').\
        replace(',', '').replace('.', '').replace('?', '').replace('(', '').replace(')', '')
    textA = [word.lower() for word in text.split()]
    textA = " ".join(textA)
    return textA

labedP["review"] = labedP["review"].apply(removeSen)
unlabedP["review"] = unlabedP["review"].apply(removeSen)

#print(labedP.head())

wordLines = pd.concat([labedP["review"], unlabedP["review"]], axis=0)
wordLines.to_csv(r"..\data\preProcessed\wordLines.txt", index=False)
labedP.to_csv(r"..\data\preProcessed\labeledData.csv", index=False)
unlabedP.to_csv(r"..\data\preProcessed\unlabeledData.csv", index=False)
