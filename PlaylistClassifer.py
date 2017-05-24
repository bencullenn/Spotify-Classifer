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
from time import time
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
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


def trainClassifer(classifier, trainFeatures, trainLabels):
    # Set variables
    clf = classifier
    le = preprocessing.LabelEncoder()

    # Normalize labels by converting label data from type string to type int
    trainLabelsData = le.fit_transform(trainLabels)
    print "Labels Transformed"

    # Save current time to variable
    start = time()

    # Train algorithm with data
    clf.fit(trainFeatures, trainLabelsData)
    print "Trained Classifier"

    # Calculate how long it took to train algorithm
    print "Runtime:", round(time() - start, 3), "s"

    # Return trained classifier
    return classifier


def testClassifer(classifier, testFeatures, testLabels):
    # Set Variables
    clf = classifier
    le = preprocessing.LabelEncoder()

    # Fit label encoder so that it can turn labels back into text
    le.fit(testLabels)

    # Use the classifier to predict labels for the test data
    predictedLabelData = clf.predict(testFeatures)

    # Convert the predicted labels back into text so they can be compared
    predictedLabelText = le.inverse_transform(predictedLabelData)

    print "Classifier Predictions:", predictedLabelText, ""

    # Calculate the accuracy of the algortihm by comparing the predicted labels to the actual labels
    print "Classifier Accuracy Score", accuracy_score(predictedLabelText, testLabels), "\n"

"""
Main Method
"""
data = pd.read_csv(filepath_or_buffer='data.csv', sep=' ')


counter = 0
while counter <= 10:
    testFeatures, testLabels, trainFeatures, trainLabels = createTestSet(data, .10)

    """
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
    """

    # Create a classifier
    print "Testing the Support Vector Machine Classifier"
    clfSVM = svm.SVC()

    # Train the algorithm
    clfSVM = trainClassifer(classifier=clfSVM, trainFeatures=trainFeatures, trainLabels=trainLabels)

    # Test the algorithm using the test data and find the accuracy
    testClassifer(classifier=clfSVM, testFeatures=testFeatures, testLabels=testLabels)

    print "Testing Stochastic Gradient Decent Classifier"
    clfSGD = SGDClassifier()
    clfSGD = trainClassifer(classifier=clfSGD, trainFeatures=trainFeatures, trainLabels=trainLabels)
    testClassifer(classifier=clfSGD, testFeatures=testFeatures, testLabels=testLabels)

    print "Testing Naive Bayes Classifier"
    clfNaiveBayes = GaussianNB()
    clfNaiveBayes = trainClassifer(classifier=clfNaiveBayes, trainFeatures=trainFeatures, trainLabels=trainLabels)
    testClassifer(classifier=clfNaiveBayes, testFeatures=testFeatures, testLabels=testLabels)

    print "Testing Decision Tree Classifier"
    clfDecisionTree = tree.DecisionTreeClassifier()
    clfDecisionTree = trainClassifer(classifier=clfDecisionTree, trainFeatures=trainFeatures, trainLabels=trainLabels)
    testClassifer(classifier=clfDecisionTree, testFeatures=testFeatures, testLabels=testLabels)

    counter += 1
