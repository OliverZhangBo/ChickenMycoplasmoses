#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 09:06
# @Author  : Arrow and Bullet
# @FileName: adaboost.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_41800366
"""Adaboost 元算法预测鸡是否患病"""
from numpy import *


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split("\t"))
    fr = open(fileName)
    dataMat = []
    labelMat = []
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        # 我觉得上面代码可以改进
        # curLine = line.strip().split("\t")
        # lineArr = curLine[0, numFeat-1]
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


#                   数据                阀值
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))  # m 行 1 列的零向量
    if threshIneq == "lt":
        retArray[dataMatrix[:, dimen] <= threshVal] = -1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10  # 用于在特征的所有可能值上进行遍历
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minError = inf  # 无穷大
    for i in range(n):  # 遍历各个特征
        rangeMin = dataMatrix[:, i].min()  # 对应特征的最小值
        rangeMax = dataMatrix[:, i].max()  # 对应特征的最大值
        stepSize = (rangeMax - rangeMin)/numSteps  # 步长
        for j in range(-1, int(numSteps) + 1):  # -1 到 numSteps
            for inequal in ["lt", "gt"]:  #
                threshVal = (rangeMin + float(j) * stepSize)  # 阀值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 这里就不是很好懂了
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print("split: dim %d, thresh %.2f, thresh inequal: %s,the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump["dim"] = i
                    bestStump["thresh"] = threshVal
                    bestStump["ineq"] = inequal
    return bestStump, minError, bestClassEst


# D = mat(ones((5, 1))/5)
# print(buildStump(dataMat, classLabels, D))


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)   # init D to all equal
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr,classLabels, D)  # build Stump
        # print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        # calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)                  # store Stump Params in Array
        # print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)  # exponent for D calc, getting messy
        D = multiply(D, exp(expon))                              # Calc New D for next iteration
        D = D/D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        # print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        # print("D:", D)
        # print("classEst:", classEst)
        # print("aggClassEst:", aggClassEst)
        print("total error:", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


# adaBoostTrainDS(dataMat, classLabels, 9)


def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)  # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])  # call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print(aggClassEst)
    return sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1, 1)
    ySum = 0
    numPosClas = sum(array(classLabels) == 1)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c="b")
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], "b--")
    plt.xlabel('False positive rate');
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum * xStep)


fileNameTr = "./data/chickMycoplasmosesTraining.txt"
fileNameTe = "./data/chickMycoplasmosesTest.txt"
dataArr, labelArr = loadDataSet(fileNameTr)

classifierArray, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)
testArr, testLabelArr = loadDataSet(fileNameTe)
prediction10 = adaClassify(testArr, classifierArray[0])
errArr = mat(ones((67, 1)))
print(errArr[prediction10 != mat(testLabelArr).T].sum())


plotROC(aggClassEst.T, labelArr)
