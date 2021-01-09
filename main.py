import RandomForest as rf

traninSet, testSet = rf.loadCSV("D:/clean_data.txt")
print(type(traninSet[0].fields))
forest = rf.buildRandomForest(traninSet)
print("准确率：", rf.test_forest(testSet,forest))
