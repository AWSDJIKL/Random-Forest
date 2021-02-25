import RandomForest as rf
import DecisionTree as dt
import time

forest_start = time.perf_counter()
traninSet, testSet = rf.loadCSV("D:/clean_data.txt")
forest = rf.buildRandomForest(traninSet)
forest_score = rf.MicroF1Measure(forest, testSet.data)
print("准确率：", rf.test_forest(testSet, forest))
forest_end = time.perf_counter()
# testData=['usual','proper','complete','one','convenient','inconv','nonprob','recommended','very_recom']
# print(testData,rf.forest_classify(testData,forest))
tree_start = time.perf_counter()
data = dt.loadCSV("D:/clean_data.txt")
testDataSet, trainDataSet = dt.SplitTestAndTrain(0.8, data)
# 根据训练集生成决策树
tree = dt.buildDecisionTree(trainDataSet)
# 剪枝
resultTree = dt.CCP(tree, testDataSet)
tree_score = dt.MicroF1Measure(resultTree, testDataSet)
tree_end = time.perf_counter()
print("森林分数:", forest_score, "运行时间:", forest_end - forest_start)
print("决策树分数:", tree_score, "运行时间", tree_end - tree_start)
