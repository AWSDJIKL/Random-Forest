import numpy as np
import copy,random,operator
# leftBranch：左子树结点
# rightBranch：右子树结点
# col：根据哪一个属性进行划分
# value：划分的值
# results：分类结果
# summary：划分结果信息（方便输出）
# data：该节点包含的样本
class Tree:
    fields = []
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




# 划分数据集样本
# dataSet：需要划分的样本
# value：划分的值（考虑换成数组，这样对于多个特征值的划分可以更有更多的划分方法）
# column：根据哪一个属性进行划分
def splitDataSet(dataSet, value, column):
    leftList = []
    rightList = []
    # 判断value是否是数值型
    if (isinstance(value, int) or isinstance(value, float)):
        # 遍历每一行数据
        for rowData in dataSet:
            # 如果某一行指定列值>=value，则将该行数据保存在leftList中，否则保存在rightList中
            if (rowData[column] >= value):
                leftList.append(rowData)
            else:
                rightList.append(rowData)
    # value为标称型
    else:
        # 遍历每一行数据
        for rowData in dataSet:
            # 如果某一行指定列值==value，则将该行数据保存在leftList中，否则保存在rightList中
            if (rowData[column] == value):
                leftList.append(rowData)
            else:
                rightList.append(rowData)
    return leftList, rightList


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
        imp += pow(results[i] / length,2)
    return 1 - imp


# 生成决策树
def buildDecisionTree(dataSet, evaluationFunc=gini):
    # 计算样本数据集的划分前的基尼指数
    baseGini = evaluationFunc(dataSet)
    # 这个样本集中已经同属一类了，无需划分，应作为叶子节点，直接返回，结束递归
    if baseGini == 0:
        tiptext = "gini:" + str(baseGini) + "   " + "result:" + str(dataSet[0][-1])
        return Tree(results=GetMajority(dataSet)[0], summary=tiptext, data=dataSet)

    # 求出总列数）
    columnLength = len(dataSet[0])
    # 计算样本集中样本总数
    rowLength = len(dataSet)
    # 初始化
    bestGini = 1  # 记录最低的gini值
    bestValue = None  # gini值最低时的列索引（划分的属性），以及划分属性的值
    bestSet = None  # 划分数据集后的数据子集
    # 标签列除外（最后一列），遍历每一列数据
    for col in range(columnLength - 1):
        # 获取指定列数据
        colSet = [example[col] for example in dataSet]
        # 获取指定列所有可能的取值（使用set函数对该列所有的值进行去重）
        uniqueColSet = set(colSet)
        # 遍历指定列每个可能的取值
        for value in uniqueColSet:
            # 分割数据集
            leftDataSet, rightDataSet = splitDataSet(dataSet, value, col)
            # 计算子数据集在源数据集的占比
            prop = len(leftDataSet) / rowLength
            # 计算划分后的gini值
            infoGini = prop * evaluationFunc(leftDataSet) + (1 - prop) * evaluationFunc(rightDataSet)
            # 记录下gini值最小时的列索引，value,数据子集
            if (bestGini > infoGini):
                bestGini = infoGini
                bestValue = (col, value)
                bestSet = (leftDataSet, rightDataSet)
    # 结点信息
    tiptext = ""
    if (isinstance(bestValue[1], int) or isinstance(bestValue[1], float)):
        tiptext = "Decision:" + str(Tree.fields[bestValue[0]]) + ">=" + str(bestValue[1]) + "   " + \
                  "gini:" + str(baseGini)
    else:
        tiptext ="Decision:" +  str(Tree.fields[bestValue[0]]) + "=" + str(bestValue[1]) + "    " + \
                  "gini:" + str(baseGini)
    leftBranch = buildDecisionTree(bestSet[0], evaluationFunc)
    rightBranch = buildDecisionTree(bestSet[1], evaluationFunc)
    return Tree(leftBranch=leftBranch, rightBranch=rightBranch, col=bestValue[0],
                value=bestValue[1], summary=tiptext, data=bestSet[0]+bestSet[1])

# 分类测试：
'''根据给定测试数据遍历二叉树，找到符合条件的叶子结点'''
def classify(data, tree):
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
        return classify(data, branch)


# 导入数据集
def loadCSV(fileName):
    # 导入数据，数据类型指定为字符串，按逗号分割
    data = np.loadtxt(fileName, dtype=str, delimiter=',')
    # 取出所有的字段
    Tree.fields = data[0]
    # 记录总样本的大小
    Tree.FullSampleSize = len(data)-1
    data = data[1:, :]
    # 转为python的list
    dataSet = data.tolist()
    return dataSet

# 打乱数据集并根据比例划分出测试数据集和训练数据集
def SplitTestAndTrain(proportion, fullDataSet):
    # 打乱顺序
    random.shuffle(fullDataSet)
    # 前面的部分为测试数据集
    testDataSet = fullDataSet[0:int(proportion * len(fullDataSet))]
    # 剩余部分为训练数据集
    trainDataSet = fullDataSet[int(proportion * len(fullDataSet)):]
    return testDataSet, trainDataSet

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

# 使用F1-Measure统计量来评价决策树
# f = 2*P*R / (1*P + R)
# P：精确率
# R：召回率
# F1越高评价越高
def MicroF1Measure(tree, testDataSet):
    # 准确分出正类，即真实值和预测值均为正类
    TP = 0
    # 错误分出正类，即真实值不为正类，但预测值为正类
    FP = 0
    # 错误分出负类，即真实值为正类，但预测值不为正类
    FN = 0
    # 测试集中所有的类别
    allClass = set([data[-1] for data in testDataSet])
    testResult=[]
    for data in testDataSet:
        # 测试集中真正的值，模型对数据的预测值
        testResult.append((data[-1], classify(data, tree)))
    for c in allClass:
        for result in testResult:
            trueValue = result[0]
            prediction = result[1]
            # print("当前正类："+str(c)+" 真实值："+str(trueValue)+" 预测值："+str(prediction))
            if trueValue == c and prediction == c:
                TP += 1
            if trueValue != c and prediction == c:
                FP += 1
            if trueValue == c and prediction != c:
                FN += 1
    # 除以类别数作总体平均
    TP = TP / len(allClass)
    FP = FP / len(allClass)
    FN = FN / len(allClass)
    # 计算精确率
    precision = TP / (TP + FP)
    # 计算召回率
    recall = TP / (TP + FN)
    F1Measure = 2 * (precision * recall) / (precision + recall)
    return F1Measure

# 对树不断递归剪枝，直到只剩一个节点
# 对所有剪枝生成的新树进行测试，选取结果最好的树
def CCP(tree, testDataSet):
    allResult={}
    allResult[copy.deepcopy(tree)] = [MicroF1Measure(tree, testDataSet),
                                      len(GetAllLeafNode(tree))+len(GetAllAlpha(tree))]
    while tree.leftBranch != None or tree.rightBranch != None:
        tree, newTree = PurningTree(tree)
        allResult[newTree] = [MicroF1Measure(newTree, testDataSet),
                              len(GetAllLeafNode(newTree))+len(GetAllAlpha(newTree))]
    #sortedResult = sorted(allResult.items(), key=lambda item: item[1], reverse=True)
    # 效果相同时，选取总节点数少的
    sortedResult = sorted(allResult.items(), key=lambda item:(item[1][0], -item[1][1]), reverse=True)
    # # 把分数都输出一次
    # for i in sortedResult:
    #     print(str(get_tree_depth(i[0]))+"  "+
    #           str(len(GetAllLeafNode(i[0]))+len(GetAllAlpha(i[0])))+" "+str(i[1]) )
    return  sortedResult[0][0]



# 选出alpha值最小的决策节点并进行剪枝，返回传进来的树和它这一次剪枝完成的副本
def PurningTree(tree):
    # 记录每个非叶子节点（即决策节点）的alpha值
    allDecisionNodes = GetAllAlpha(tree)
    # 进行排序，选出最小的进行剪枝
    sortedResult = sorted(allDecisionNodes.items(), key=lambda item: item[1], reverse=False)
    node = sortedResult[0][0]
    #print("这次剪去的节点的高度是："+str(get_tree_depth(node)))
    # 让该节点成为叶子节点
    node.leftBranch = None
    node.rightBranch = None
    majority, number = GetMajority(node.data)
    bestGini = gini(node.data)
    node.results = majority
    tiptext = "gini:" + str(bestGini) + "   " + "result:" + str(majority)
    node.summary = tiptext
    coptTree = copy.deepcopy(tree)
    return tree, coptTree



# 计算整棵树所有非叶子节点的alpha值
def GetAllAlpha(tree):
    allDecisionNodes = {}
    # 是叶子节点，直接返回
    if tree.leftBranch == None and tree.rightBranch == None:
        return {}
    if tree.leftBranch:
        allDecisionNodes.update(GetAllAlpha(tree.leftBranch))
    if tree.rightBranch:
        allDecisionNodes.update(GetAllAlpha(tree.rightBranch))
    allDecisionNodes[tree] = Alpha(tree)
    return allDecisionNodes

# 计算该子树的alpha值(代表若对该节点进行剪枝所带来的代价和减少的复杂度)
# alpha = (节点的错误代价 - 整个子树的错误代价) / (该子树的叶子节点数 - 1)
def Alpha(tree):
    nodeCost = NodeCost(tree)
    treeCost = TreeCost(tree)
    leafCount = len(GetAllLeafNode(tree))
    alpha = (nodeCost - treeCost) / (leafCount - 1)
    return alpha

# 计算子树的错误代价
# 子树的错误代价 = 所有叶子节点的错误代价之和
def TreeCost(tree):
    leaves = GetAllLeafNode(tree)
    cost = 0
    for leaf in leaves:
        cost += NodeCost(leaf)
    return cost

# 计算节点的错误代价
# 节点的错误代价 = 错分样本占比 * 节点样本在总体样本的占比
def NodeCost(tree):
    # 找出这个节点代表的类型
    majorty, majortySize = GetMajority(tree.data)
    # 计算该节点的样本大小
    nodeSize=len(tree.data)
    cost = ((nodeSize - majortySize) / nodeSize) * (nodeSize / Tree.FullSampleSize)
    return cost


# 获取所有的叶子节点
def GetAllLeafNode(tree):
    leaves = []
    if tree.leftBranch:
        leaves.extend(GetAllLeafNode(tree.leftBranch))
    if tree.rightBranch:
        leaves.extend(GetAllLeafNode(tree.rightBranch))
    if tree.leftBranch == None and tree.rightBranch == None:
        leaves.append(tree)
    return leaves


#获取决策树的深度
def get_tree_depth(tree):
    depth = 1
    leftDepth=0
    rightDepth=0
    # 是叶子节点
    if tree.leftBranch == None and tree.rightBranch == None:
        return 1
    # 计算左子树的叶子节点
    if tree.leftBranch:
        leftDepth = get_tree_depth(tree.leftBranch)
    # 计算右子树的叶子节点
    if tree.rightBranch:
        rightDepth = get_tree_depth(tree.rightBranch)
    return leftDepth + 1 if leftDepth > rightDepth else rightDepth + 1
