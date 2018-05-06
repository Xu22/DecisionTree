#coding:utf-8
# ID3算法，建立决策树
import numpy as np
# import math
# # import uniout
# '''
#  #创建数据集
#  def creatDataSet():
#      dataSet = np.array([[1,1,'yes'],
#                          [1,1,'yes'],
#                          [1,0,'no'],
#                          [0,1,'no'],
#                          [0,1,'no']])
#      features = ['no surfaceing', 'fippers']
#      return dataSet, features
# '''
# #创建数据集
# def createDataSet():
#      dataSet = np.array([['青年', '否', '否', '否'],
#                    ['青年', '否', '否', '否'],
#                    ['青年', '是', '否', '是'],
#                    ['青年', '是', '是', '是'],
#                    ['青年', '否', '否', '否'],
#                    ['中年', '否', '否', '否'],
#                    ['中年', '否', '否', '否'],
#                    ['中年', '是', '是', '是'],
#                    ['中年', '否', '是', '是'],
#                    ['中年', '否', '是', '是'],
#                    ['老年', '否', '是', '是'],
#                    ['老年', '否', '是', '是'],
#                    ['老年', '是', '否', '是'],
#                    ['老年', '是', '否', '是'],
#                    ['老年', '否', '否', '否']])
#      features = ['年龄', '有工作', '有女朋友']
#      return dataSet, features
# #计算数据集的熵
# def calcEntropy(dataSet):
#      #先算概率
#      labels = list(dataSet[:,-1])
#      # print(len(labels))
#      prob = {}
#      entropy = 0.0
#      for label in labels:
#          prob[label] = (labels.count(label) / float(len(labels)))
#          # print(labels.count(label))
#      for v in prob.values():
#          entropy = entropy + (-v * math.log(v,2))
#          # print(entropy)
#      return entropy
# #划分数据集
# def splitDataSet(dataSet, i, fc):
#      subDataSet = []
#      for j in range(len(dataSet)):
#          if dataSet[j, i] == str(fc):
#              sbs = []
#              sbs.append(dataSet[j, :])
#              subDataSet.extend(sbs)
#      subDataSet = np.array(subDataSet)
#      return np.delete(subDataSet,[i],1)
#  #计算信息增益，选择最好的特征划分数据集，即返回最佳特征下标
# def chooseBestFeatureToSplit(dataSet):
#      labels = list(dataSet[:, -1])
#      bestInfoGain = 0.0   #最大的信息增益值
#      bestFeature = -1   #*******
#      #摘出特征列和label列
#      for i in range(dataSet.shape[1]-1):     #列
#          #计算列中，各个分类的概率
#          prob = {}
#          featureCoulmnL = list(dataSet[:,i])
#          for fcl in featureCoulmnL:
#              prob[fcl] = featureCoulmnL.count(fcl) / float(len(featureCoulmnL))
#          #计算列中，各个分类的熵
#          new_entrony = {}    #各个分类的熵
#          condi_entropy = 0.0   #特征列的条件熵
#          featureCoulmn = set(dataSet[:,i])   #特征列
#          for fc in featureCoulmn:
#              subDataSet = splitDataSet(dataSet, i, fc)
#              prob_fc = len(subDataSet) / float(len(dataSet))
#              new_entrony[fc] = calcEntropy(subDataSet)   #各个分类的熵
#              condi_entropy = condi_entropy + prob[fc] * new_entrony[fc]    #特征列的条件熵
#          infoGain = calcEntropy(dataSet) - condi_entropy     #计算信息增益
#          if infoGain > bestInfoGain:
#              bestInfoGain = infoGain
#              bestFeature = i
#      return bestFeature
#  #若特征集features为空，则T为单节点，并将数据集D中实例树最大的类label作为该节点的类标记，返回T
# def majorityLabelCount(labels):
#      labelCount = {}
#      for label in labels:
#          if label not in labelCount.keys():
#              labelCount[label] = 0
#          labelCount[label] += 1
#      return max(labelCount)
#  #建立决策树T
# def createDecisionTree(dataSet, features):
#      labels = list(dataSet[:,-1])
#      # print(dataSet[0])
#      #如果数据集中的所有实例都属于同一类label，则T为单节点树，并将类label作为该结点的类标记，返回T
#      if len(set(labels)) == 1:  #set是一个无序且不重复的元素集合
#          return labels[0]
#      #若特征集features为空，则T为单节点，并将数据集D中实例树最大的类label作为该节点的类标记，返回T
#      if len(dataSet[0]) == 1:
#          return majorityLabelCount(labels)
#      #否则，按ID3算法就计算特征集中各特征对数据集D的信息增益，选择信息增益最大的特征beatFeature
#      bestFeatureI = chooseBestFeatureToSplit(dataSet)  #最佳特征的下标
#      bestFeature = features[bestFeatureI]    #最佳特征
#      decisionTree = {bestFeature:{}} #构建树，以信息增益最大的特征beatFeature为子节点
#      del(features[bestFeatureI])    #该特征已最为子节点使用，则删除，以便接下来继续构建子树
#      bestFeatureColumn = set(dataSet[:,bestFeatureI])
#      for bfc in bestFeatureColumn:
#          subFeatures = features[:]
#          decisionTree[bestFeature][bfc] = createDecisionTree(splitDataSet(dataSet, bestFeatureI, bfc), subFeatures)
#      return decisionTree
#  #对测试数据进行分类
# def classify(testData, features, decisionTree):
#      for key in decisionTree:
#          index = features.index(key)
#          testData_value = testData[index]
#          subTree = decisionTree[key][testData_value]
#          if type(subTree) == dict:
#              result = classify(testData,features,subTree)
#              return result
#          else:
#              return subTree
# if __name__ == '__main__':
#      dataSet, features = createDataSet()     #创建数据集
#      decisionTree = createDecisionTree(dataSet, features)   #建立决策树
#      print(decisionTree)
#      dataSet, features = createDataSet()
#      testData = ['老年', '是', '否']
#      result = classify(testData, features, decisionTree)  #对测试数据进行分类
#      print('是否给:%s,贷款:%s'%(testData,result))
import pandas as pd
f=open("sample_submission.csv")
dataf=pd.read_csv(f)
x=dataf.iloc[1:,1:5].as_matrix()
y=dataf.iloc[1:,5].as_matrix()
print(len(x))
for i in range(0,len(x)):
    for j in range(0,len(x[i])):
        thisdata=x[i][j]
        if(thisdata=='是'or thisdata=='多' or thisdata=='高'):
            x[i][j]=int(1)
        else:
            x[i][j]=-1
for i in range(0,len(y)):
    thisdata=y[i]
    if(thisdata=='否' or thisdata=='少' or thisdata=='低'):
        y[i]=int(1)
    else:
        y[i]=-1
#转化好格式，将x、y转化为数据框，然后再转化为数组并指定格式
xf=pd.DataFrame(x)
yf=pd.DataFrame(y)
x2=xf.as_matrix().astype(int)
y2=yf.as_matrix().astype(int)
from sklearn.tree import DecisionTreeClassifier as DTC
dtc=DTC(criterion='entropy')
dtc.fit(x2,y2)
#直接预测销量高低
import numpy as np
x3=np.array([[1,-1,-1,1],[1,1,1,1],[-1,1,-1,1]])
result=dtc.predict(x3)
print(result)
#可视化决策树
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
with open('d:\\test.dot','w') as file:
    export_graphviz(dtc,feature_names= ['combat','num','promotion','datum'],out_file= file)
# dot -Tpng test.dot -o lesson.png
