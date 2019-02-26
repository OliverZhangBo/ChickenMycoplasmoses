#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/11/26 10:27
# @Author  : Arrow and Bullet
# @FileName: logRegression.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_41800366

"""Logistic 回归预测鸡是否患病"""


from numpy import *


# 加载了数据集
def loadDataSet(fileName):
    dataMat = []  # 数据
    labelMat = []  # 标签
    fr = open(fileName)  # 创建fr文件对象
    for line in fr.readlines():  # 读文件
        currLine = line.strip().split("\t")  # 每一行做一个数组
        lenCurrLine = len(currLine)
        lineArr = []
        for i in range(lenCurrLine):
            lineArr.append(float(currLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))  # 标签
    return dataMat, labelMat


def sigmoid(inX):
    return 1/(1+exp(-inX))  # 实现sigmoid函数


# stochastic /sto'kæstɪk/ 随机的；猜测的
def stocGradAscent(dataMatrix, classLabels, nummIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(nummIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1+j+i)+0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha*error*dataMatrix[randIndex]
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def chickMycoplasmosesTest(fileNameTr, fileNameTe):
    frTest = open(fileNameTe)
    trainingSet, trainingLabels = loadDataSet(fileNameTr)
    trainingWeights = stocGradAscent(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split("\t")
        lenCurrLine = len(currLine)
        lineArr = []
        for i in range(lenCurrLine):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainingWeights)) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest(fileNameTr, fileNameTe):
    numTests = 10
    errorSum = 0
    for k in range(numTests):
        errorSum += chickMycoplasmosesTest(fileNameTr, fileNameTe)
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))


fileNameTr = "./data/chickMycoplasmosesTraining.txt"
fileNameTe = "./data/chickMycoplasmosesTest.txt"
multiTest(fileNameTr, fileNameTe)