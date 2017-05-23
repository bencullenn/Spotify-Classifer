"""
This class takes two text files of JSON Formatted data for a Spotify playlist and 
trains a classification algorithm based on the data in the "Training Set" text file. 

The algorithm is then tested with the "Test Set" text file and the accuracy score is output. 
The algorithm can then be used give the fit of a link to any song on Spotify
"""

import pandas as pd
import random
import sklearn

def createTestSet(dataSet, testSetProportion):
    testSetSize = int(len(dataSet)*testSetProportion)
    trainSize = int(len(dataSet) * (1-testSetProportion))
    trainSet = []
    testSet = []
    for i, row in dataSet.iterrows():  # i: dataframe index; row: each row in series format
        if i % 10 == 0:
            testSet.append(row)
        else:
            trainSet.append(row)

    print ("{}{}".format("DataSet Size:", len(dataSet)))
    print ("{}{}".format("TestSet Size:", len(testSet)))
    print ("{}{}".format("TrainSet Size:", len(trainSet)))
    return testSet, trainSet

data = pd.read_csv(filepath_or_buffer='data.csv', sep=' ')
print (data)
test, train = createTestSet(data, .10)

for p in test:
    print(p)

for p in train:
    print(p)

