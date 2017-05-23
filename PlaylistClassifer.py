"""
This class takes two text files of JSON Formatted data for a Spotify playlist and 
trains a classification algorithm based on the data in the "Training Set" text file. 

The algorithm is then tested with the "Test Set" text file and the accuracy score is output. 
The algorithm can then be used give the fit of a link to any song on Spotify
"""

import pandas as pd
import numpy as np
import random
from sklearn import svm
import pprint

def createTestSet(dataSet, testSetProportion):
    testSize = int(len(dataSet)*testSetProportion)
    trainSize = int(len(dataSet) * (1-testSetProportion))
    trainSetFeatures = np.empty(6, dtype=float)
    trainSetLabels = np.empty(1, dtype=str)

    testSetFeatures = np.empty(6, dtype=float)
    testSetLabels = np.empty(1, dtype=str)


    for i, row in dataSet.iterrows():  # i: dataframe index; row: each row in series format

        rowData = row.values

        rowFeatures = rowData[1:7]
        rowLabel = np.empty((0, 1), dtype=str)
        rowLabel = np.append(rowLabel, rowData[7])

        if i % 10 == 0:
            np.append(testSetFeatures, rowFeatures, axis=0)
            np.append(testSetLabels, rowLabel, axis=0)
            
        else:
            np.append(trainSetFeatures, rowFeatures, axis=0)
            np.append(trainSetLabels, rowLabel, axis=0)

    print ("{}{}".format("DataSet Size:", len(dataSet)))
    print ("{}{}".format("TestSetLabels Size:", len(testSetLabels)))
    print ("{}{}".format("TestSetFeatures Size:", len(testSetFeatures)))
    print ("{}{}".format("TrainSetLabels Size:", len(trainSetLabels)))
    print ("{}{}".format("TrainSetFeatures Size:", len(trainSetFeatures)))

    return testSetFeatures, testSetLabels, trainSetFeatures, trainSetLabels

data = pd.read_csv(filepath_or_buffer='data.csv', sep=' ')

testFeatures, testLabels, trainFeatures, trainLabels = createTestSet(data, .10)


print "Train Features"
pprint.pprint(trainFeatures)

print "Train Labels"
pprint.pprint(trainLabels)


print "Test Features"
pprint.pprint(testFeatures)

print "Test Labels"
pprint.pprint(testLabels)


