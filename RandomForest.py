import numpy as np
import copy, random, operator

class Tree:
    FullSampleSize = 0
    def __init__(self, leftBranch=None, rightBranch=None, col=-1, value=None, results=None, summary=None, data=None):
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.col = col
        self.value = value
        self.results = results
        self.summary = summary
        self.data = data


    def ToDict(self):
        dict={}
        dict["summary"] = self.summary
        dict["results"] = self.results
        if self.leftBranch:
            dict["leftBranch"] = Tree.ToDict(self.leftBranch)
        if self.rightBranch:
            dict["rightBranch"] = Tree.ToDict(self.rightBranch)
        return dict

class DataSet:
    fields = {}
    data = []
    def __init__(self, fields={}, data=[]):
        self.fields = fields
        self.data = data


# 导入数据集，随机生成N个样本
def loadCSV(fileName):
    dataSets = []
    # 导入数据，数据类型指定为字符串，按逗号分割
    data = np.loadtxt(fileName, dtype=str, delimiter=',')
    # 建立特征池
    fields = {}
    for i in range(len(data[0])):
        fields[data[0][i]] = i
    # 记录总样本的大小
    Tree.FullSampleSize = len(data)-1
    # 首行为列名
    data = data[1:, :]
    # 转为python的list
    data = data.tolist()
    # 随机划分出N+1个样本集，每个样本集里由N个样本，从原来的样本里有放回式抽取
    for i in range(len(data)):
    # for i in range(10):
        print("1")
        dataSet = DataSet()
        for j in range(len(data)):
            dataSet.data.append(data[random.randint(0, len(data) - 1)])
        # 特征池
        dataSet.fields = copy.deepcopy(fields)
        dataSets.append(dataSet)
    # 第一个作为测试集，后N个作为训练集
    return dataSets[1:], dataSets[0]

# 划分数据集样本
# dataSet：需要划分的样本
# value：划分的值
# column：根据哪一个属性进行划分
def splitDataSet(dataSet, value, column):
    leftList = []
    rightList = []
    # 判断value是否是数值型
    if (isinstance(value, int) or isinstance(value, float)):
        # 遍历每一行数据
        for rowData in dataSet.data:
            # 如果某一行指定列值>=value，则将该行数据保存在leftList中，否则保存在rightList中
            if (rowData[column] >= value):
                leftList.append(rowData)
            else:
                rightList.append(rowData)
    # value为标称型
    else:
        # 遍历每一行数据
        for rowData in dataSet.data:
            # 如果某一行指定列值==value，则将该行数据保存在leftList中，否则保存在rightList中
            if (rowData[column] == value):
                leftList.append(rowData)
            else:
                rightList.append(rowData)
    leftSet = DataSet(copy.deepcopy(dataSet.fields), leftList)
    rightSet = DataSet(copy.deepcopy(dataSet.fields), rightList)
    return leftSet, rightSet

# 统计标签类每个样本个数
'''
该函数是计算gini值的辅助函数，假设输入的dataSet为为['A', 'B', 'C', 'A', 'A', 'D']，
则输出为['A':3,' B':1, 'C':1, 'D':1]，这样分类统计dataSet中每个类别的数量
'''

def calculateDiffCount(dataSet):
    results = {}
    for data in dataSet:
        # data[-1] 是数据集最后一列，也就是标签类
        if data[-1] not in results:
            results.setdefault(data[-1], 1)
        else:
            results[data[-1]] += 1
    return results


# 计算基尼值
def gini(dataSet):
    # 计算样本总数
    length = len(dataSet)
    # 计算数据集中各个类别的数量
    results = calculateDiffCount(dataSet)
    imp = 0.0
    for i in results:
        imp += pow(results[i] / length, 2)
    return 1 - imp

def buildRandomForest(dataSets):
    forest = []
    for i in range(len(dataSets)):
        forest.append(buildDecisionTree(dataSets[i]))
    return forest

# 根据给定样本集生成决策树
def buildDecisionTree(dataSet, evaluationFunc=gini):
    print("2")
    print(type(dataSet.data))
    print(type(dataSet.fields))
    # 计算样本数据集的划分前的基尼指数
    baseGini = evaluationFunc(dataSet.data)
    # 这个样本集中已经同属一类了，无需划分，应作为叶子节点，直接返回，结束递归
    # 若所有特征都已用过，则也返回
    if baseGini == 0 or len(dataSet.fields) == 0:
        tiptext = "gini:" + str(baseGini) + "   " + "result:" + str(GetMajority(dataSet.data)[0])
        return Tree(results=GetMajority(dataSet.data)[0], summary=tiptext)
    # 计算样本集中样本总数
    rowLength = len(dataSet.data)
    # 初始化
    bestGini = 1  # 记录最低的gini值
    bestValue = None  # gini值最低时的列索引（划分的属性），以及划分属性的值
    bestSet = None  # 划分数据集后的数据子集
    # 从特征池里随机选取其中一个特征，然后在该特征中所有可能的取值中选取最佳分割点
    columnName = random.sample(dataSet.fields.keys(), 1)
    column = dataSet.fields.pop(columnName[0])
    # 获取指定列数据
    colSet = [example[column] for example in dataSet.data]
    # 获取指定列所有可能的取值（使用set函数对该列所有的值进行去重）
    uniqueColSet = set(colSet)
    # 遍历指定列每个可能的取值
    for value in uniqueColSet:
        # 分割数据集
        leftDataSet, rightDataSet = splitDataSet(dataSet, value, column)
        # 计算子数据集在源数据集的占比
        prop = len(leftDataSet.data) / rowLength
        # 计算划分后的gini值
        infoGini = prop * evaluationFunc(leftDataSet.data) + (1 - prop) * evaluationFunc(rightDataSet.data)
        # 记录下gini值最小时的列索引，value,数据子集
        if (bestGini > infoGini):
            bestGini = infoGini
            bestValue = (column, value)
            bestSet = (leftDataSet, rightDataSet)
    # 结点信息
    tiptext = ""
    if (isinstance(bestValue[1], int) or isinstance(bestValue[1], float)):
        tiptext = "Decision:" + str(columnName) + ">=" + str(bestValue[1]) + "   " + \
                  "gini:" + str(baseGini)
    else:
        tiptext ="Decision:" +  str(columnName) + "=" + str(bestValue[1]) + "    " + \
                  "gini:" + str(baseGini)
    leftBranch = buildDecisionTree(bestSet[0], evaluationFunc)
    rightBranch = buildDecisionTree(bestSet[1], evaluationFunc)
    return Tree(leftBranch=leftBranch, rightBranch=rightBranch, col=bestValue[0],
                value=bestValue[1], summary=tiptext)

# 分类测试：
'''根据给定测试数据遍历二叉树，找到符合条件的叶子结点'''
def tree_classify(data, tree):
    # 判断是否是叶子结点，是就返回叶子结点相关信息，否就继续遍历
    if tree.results != None:
        return tree.results
    else:
        branch = None
        v = data[tree.col]
        # 数值型数据
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.leftBranch
            else:
                branch = tree.rightBranch
        else:  # 标称型数据
            if v == tree.value:
                branch = tree.leftBranch
            else:
                branch = tree.rightBranch
        return tree_classify(data, branch)

# 森林投票，输出得票最多的结果
def forest_classify(data, forest):
    result = []
    for i in range(len(forest)):
        print("3")
        result.append(tree_classify(data, forest[i]))
    return max(result, key=result.count)


def test_forest(testSet, forest):
    correct = 0
    for i in range(len(testSet.data)):
        result = forest_classify(testSet.data[i], forest)
        if result == testSet.data[i][-1]:
            correct += 1
    # 返回预测准确率
    return correct / len(testSet.data)

# 返回给定的样本集中最多的类型以及数量
def GetMajority(dataSet):
    # 记录所有的类型的数量
    classCounts = {}
    for value in dataSet:
        #print(value[-1])
        if (value[-1] not in classCounts.keys()):
            classCounts[value[-1]] = 0
        classCounts[value[-1]] += 1
    sortedClassCount = sorted(classCounts.items(), key=lambda item: item[1], reverse=True)
    return sortedClassCount[0][0], sortedClassCount[0][1]



