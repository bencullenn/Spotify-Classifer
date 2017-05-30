"""
This class takes two text files of JSON Formatted data for a Spotify playlist and 
trains a classification algorithm based on the data in the "Training Set" text file. 

The algorithm is then tested with the "Test Set" text file and the accuracy score is output. 
The algorithm can then be used give the fit of a link to any song on Spotify
"""

import pandas as pd
import numpy as np
from time import time
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import sys
import spotipy
import spotipy.util as util
import pprint

"""
Functions
"""


def createTestSet(dataSet, testSetProportion):

    testSize = int(len(dataSet)*testSetProportion)
    trainSize = int(len(dataSet) * (1-testSetProportion))

    # Create 2D arrays to store test and training sets
    trainSetFeatures = np.empty((0, 7), dtype=float)
    trainSetLabels = np.empty((0, 1), dtype=str)

    testSetFeatures = np.empty((0, 7), dtype=float)
    testSetLabels = np.empty((0, 1), dtype=str)

    for i, row in dataSet.iterrows():  # i: dataframe index; row: each row in series format

        # Convert the row data into an array
        rowData = row.values

        # Extract feature data into an array
        rowFeatures = rowData[0:7]

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
    pprint.pprint(trainSetFeatures)
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
    # Calculate how long it took to train algorithm
    runtime = time() - start
    print "Trained Classifier"
    print "Runtime:", round(runtime, 3), "s"

    # Return trained classifier and runtime
    return classifier, runtime


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
    accuracy = accuracy_score(predictedLabelText, testLabels)
    print "Classifier Accuracy Score", accuracy, "\n"

    # Return accuracy
    return accuracy


def testClassifers(amountOfTests, data):
    # Create counter variable for loop
    counter = 1

    # Create variables to store the best values from testing
    bestRuntimeSVM = 100.00000
    topAccuracySVM = 0.00000

    bestRuntimeSGD = 100.00000
    topAccuracySGD = 0.00000

    bestRuntimeNaiveBayes = 100.00000
    topAccuracyNaiveBayes = 0.00000

    bestRuntimeDecTree = 100.00000
    topAccuracyDecTree = 0.00000

    while counter <= amountOfTests:
        print "Test Number ", counter
        testFeatures, testLabels, trainFeatures, trainLabels = createTestSet(data, .10)

        # Create a classifier
        print "Testing the Support Vector Machine Classifier"
        clfSVM = svm.SVC()

        # Train the algorithm
        clfSVM, sessionRuntimeSVM = trainClassifer(classifier=clfSVM, trainFeatures=trainFeatures,
                                                   trainLabels=trainLabels)

        # Test the algorithm using the test data and find the accuracy
        sessionAccuracySVM = testClassifer(classifier=clfSVM, testFeatures=testFeatures, testLabels=testLabels)

        print "Testing Stochastic Gradient Decent Classifier"
        clfSGD = SGDClassifier()
        clfSGD, sessionRuntimeSGD = trainClassifer(classifier=clfSGD, trainFeatures=trainFeatures,
                                                   trainLabels=trainLabels)
        sessionAccuracySGD = testClassifer(classifier=clfSGD, testFeatures=testFeatures, testLabels=testLabels)

        print "Testing Naive Bayes Classifier"
        clfNaiveBayes = GaussianNB()
        clfNaiveBayes, sessionRuntimeNaiveBayes = trainClassifer(classifier=clfNaiveBayes, trainFeatures=trainFeatures,
                                                                 trainLabels=trainLabels)
        sessionAccuracyNaiveBayes = testClassifer(classifier=clfNaiveBayes, testFeatures=testFeatures,
                                                  testLabels=testLabels)

        print "Testing Decision Tree Classifier"
        clfDecisionTree = tree.DecisionTreeClassifier()
        clfDecisionTree, sessionRuntimeDecTree = trainClassifer(classifier=clfDecisionTree, trainFeatures=trainFeatures,
                                                                trainLabels=trainLabels)
        sessionAccuracyDecTree = testClassifer(classifier=clfDecisionTree, testFeatures=testFeatures,
                                               testLabels=testLabels)

        # If any of the session values are better than the best value update the best value
        if sessionRuntimeSVM < bestRuntimeSVM:
            bestRuntimeSVM = sessionRuntimeSVM
        if sessionAccuracySVM > topAccuracySVM:
            topAccuracySVM = sessionAccuracySVM

        if sessionRuntimeSGD < bestRuntimeSGD:
            bestRuntimeSGD = sessionRuntimeSGD
        if sessionAccuracySGD > topAccuracySGD:
            topAccuracySGD = sessionAccuracySGD

        if sessionRuntimeNaiveBayes < bestRuntimeNaiveBayes:
            bestRuntimeNaiveBayes = sessionRuntimeNaiveBayes
        if sessionAccuracyNaiveBayes > topAccuracyNaiveBayes:
            topAccuracyNaiveBayes = sessionAccuracyNaiveBayes

        if sessionRuntimeDecTree < bestRuntimeDecTree:
            bestRuntimeDecTree = sessionRuntimeDecTree
        if sessionAccuracyDecTree > topAccuracyDecTree:
            topAccuracyDecTree = sessionAccuracyDecTree

        counter += 1

    print "Support Vector Machine Classifier Results"
    print "Best Runtime", bestRuntimeSVM
    print "Top Accuracy", topAccuracySVM

    print "Stochastic Gradient Decent Classifier Results"
    print "Best Runtime", bestRuntimeSGD
    print "Top Accuracy", topAccuracySGD

    print "Naive Bayes Classifier Results"
    print "Best Runtime", bestRuntimeNaiveBayes
    print "Top Accuracy", topAccuracyNaiveBayes

    print "Decision Tree Classifier Results"
    print "Best Runtime", bestRuntimeDecTree
    print "Top Accuracy", topAccuracyDecTree

    topClassiferAccuracy = topAccuracySVM
    mostAccurateClassifier = clfSVM

    if (topAccuracySGD > topClassiferAccuracy):
        topClassiferAccuracy = topAccuracySGD
        mostAccurateClassifier = clfSGD

    if (topAccuracyNaiveBayes > topClassiferAccuracy):
        topClassiferAccuracy = topAccuracyNaiveBayes
        mostAccurateClassifier = clfNaiveBayes

    if (topClassiferAccuracy > topAccuracyDecTree):
        topClassiferAccuracy = topAccuracyDecTree
        mostAccurateClassifier = topAccuracyDecTree

    return mostAccurateClassifier

def parseTrackLink(trackLink):
    isLinkValad = False
    trackID = ""

    linkInfo = trackLink[25:]
    print "Link info:", linkInfo

    if linkInfo[:5] == "track":
        isLinkValad = True
        print "Link belongs to a track"
        trackID = linkInfo[6:]

        print "Track ID:", trackID
    else:
        print "Link does not belong to a track"

    return isLinkValad, trackID

def getUsername():
    if len(sys.argv) > 1:
        # Ask the user for their username
        return sys.argv[1]
    else:
        # If username cannot be found then throw an error
        print "Usage: %s username" % (sys.argv[0],)
        sys.exit()


def createTokenForScope(username, scope):
    # Get token for user
    token = util.prompt_for_user_token(username, scope)

    # If given valid username token
    if token:
        print ("Token Successfully Generated")
        return token
    else:
        print "Can't get token for", username


"""
Main Method
"""

data = pd.read_csv(filepath_or_buffer='data.csv', sep=' ')
mostAccurateClassifier = testClassifers(amountOfTests=1, data=data)

isTrackLinkValad, trackID = parseTrackLink("https://open.spotify.com/track/4MUyxhxNFRViaJzJYQoYqE")

scope = 'user-read-private user-read-email playlist-modify-private playlist-modify-public'

# Create username and Token objects
authorizationUsername = getUsername()
token = createTokenForScope(scope=scope, username=authorizationUsername)

# Create a Spotipy object
sp = spotipy.Spotify(auth=token)

trackData = sp.track(trackID)
trackAudioFeatures = sp.audio_features([trackID])[0]

trackFeatures = np.empty((0, 7), dtype=float)
trackFeatures = np.append(trackFeatures, [trackAudioFeatures['acousticness']])
trackFeatures = np.append(trackFeatures, [trackAudioFeatures['danceability']])
trackFeatures = np.append(trackFeatures, [trackAudioFeatures['energy']])
trackFeatures = np.append(trackFeatures, [trackAudioFeatures['instrumentalness']])
trackFeatures = np.append(trackFeatures, [trackAudioFeatures['liveness']])
trackFeatures = np.append(trackFeatures, [trackAudioFeatures['speechiness']])
trackFeatures = np.append(trackFeatures, [trackAudioFeatures['valence']])

prediction = mostAccurateClassifier.predict(trackFeatures)

print prediction
