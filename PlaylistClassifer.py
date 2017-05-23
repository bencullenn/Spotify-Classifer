"""
This class takes two text files of JSON Formatted data for a Spotify playlist and 
trains a classification algorithm based on the data in the "Training Set" text file. 

The algorithm is then tested with the "Test Set" text file and the accuracy score is output. 
The algorithm can then be used give the fit of a link to any song on Spotify
"""

import pandas as pd
import numpy as np
import random
import sklearn
import pprint
from sklearn import preprocessing
from sklearn import svm

"""
Functions
"""
def createTestSet(dataSet, testSetProportion):

    testSize = int(len(dataSet)*testSetProportion)
    trainSize = int(len(dataSet) * (1-testSetProportion))

    # Create 2D arrays to store test and training sets
    trainSetFeatures = np.empty((0, 6), dtype=float)
    trainSetLabels = np.empty((0, 1), dtype=str)

    testSetFeatures = np.empty((0, 6), dtype=float)
    testSetLabels = np.empty((0, 1), dtype=str)

    for i, row in dataSet.iterrows():  # i: dataframe index; row: each row in series format

        # Convert the row data into an array
        rowData = row.values

        # Extract feature data into an array
        rowFeatures = rowData[1:7]

        # Extract the label and insert it into an array
        rowLabel = np.empty((0, 1), dtype=str)
        rowLabel = np.append(rowLabel, rowData[7])

        # Every 10 rows
        if i % 10 == 0:

            # Take the row features and label and add it to the test set
            # print ("{}{}".format("Features added to Test Set", rowFeatures))
            testSetFeatures = np.append(testSetFeatures, [rowFeatures], axis=0)
            # print ("{}{}".format("Label added to Test Set:", rowLabel))
            testSetLabels = np.append(testSetLabels, [rowLabel])
            
        else:

            # Take the row features and label and add it to the training set
            # print ("{}{}".format("Features added to Train Set", rowFeatures))
            trainSetFeatures = np.append(trainSetFeatures, [rowFeatures], axis=0)
            # print ("{}{}".format("Label added to Train Set:", rowLabel))
            trainSetLabels = np.append(trainSetLabels, [rowLabel])

    # Print out some data to verify that the training and test sets are proper size
    print ("{}{}".format("DataSet Size:", len(dataSet)))
    print ("{}{}".format("TestSetLabels Size:", len(testSetLabels)))
    print ("{}{}".format("TestSetFeatures Size:", len(testSetFeatures)))
    print ("{}{}".format("TrainSetLabels Size:", len(trainSetLabels)))
    print ("{}{}".format("TrainSetFeatures Size:", len(trainSetFeatures)))
    print "\n"
    return testSetFeatures, testSetLabels, trainSetFeatures, trainSetLabels

"""
Code
"""
data = pd.read_csv(filepath_or_buffer='data.csv', sep=' ')

testFeatures, testLabels, trainFeatures, trainLabels = createTestSet(data, .10)


print "Train Features"
print(trainFeatures)
print "\n"

print "Train Labels"
print(trainLabels)
print "\n"

print "Test Features"
print(testFeatures)
print "\n"

print "Test Labels"
print(testLabels)
print "\n"

# Normalize labels

le = preprocessing.LabelEncoder()

trainLabels = le.fit_transform(trainLabels)
print "Labels Transformed\n"

# Time to train some Classifers

svmClf = svm.SVC()

svmClf.fit(trainFeatures, trainLabels)
print "Support Vector Machine Trained\n"

# Calculate Accuracy

svmPredictionData = svmClf.predict(testFeatures)
svmPredictions = le.inverse_transform(svmPredictionData)

print "{}{}".format("SVM Predictions", svmPredictions)
