"""
This class takes two text files of JSON Formatted data for a Spotify playlist and 
trains a classification algorithm based on the data in the "Training Set" text file. 

The algorithm is then tested with the "Test Set" text file and the accuracy score is output. 
The algorithm can then be used give the fit of a link to any song on Spotify
"""

import pandas as pd
import numpy as np
import sys
import spotipy
import spotipy.util as util
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

#need to randomize creating test set so it's not the same each time. Use shufflesplit
#print out
def create_test_set(data_set, test_set_proportion):

    test_size = int(len(data_set) * test_set_proportion)
    train_size = int(len(data_set) * (1 - test_set_proportion))

    # Create 2D arrays to store test and training sets.
    # features array stores characteristics like acousticness and labels stores whether it is "in" or "out" of the positive examples playlist
    train_set_features = np.empty((0, 7), dtype=float)
    train_set_labels = np.empty((0, 1), dtype=str)

    test_set_features = np.empty((0, 7), dtype=float)
    test_set_labels = np.empty((0, 1), dtype=str)

    for i, row in data_set.iterrows():  # i: dataframe index; row: each row in series format

        # Convert the row data into an array
        row_data = row.values

        # Extract feature data into an array
        row_features = row_data[0:7]

        # Extract the label and insert it into an array
        row_label = np.empty((0, 1), dtype=str)
        row_label = np.append(row_label, row_data[7])

        # Every 10 rows
        if i % 10 == 0:

            # Take the row features and label and add it to the test set
            test_set_features = np.append(test_set_features, [row_features], axis=0)
            test_set_labels = np.append(test_set_labels, [row_label])
            
        else:

            # Take the row features and label and add it to the training set
            train_set_features = np.append(train_set_features, [row_features], axis=0)
            train_set_labels = np.append(train_set_labels, [row_label])

    # Print out some data to verify that the training and test sets are proper size
    print ("{}{}".format("DataSet Size:", len(data_set)))
    print ("{}{}".format("TestSetLabels Size:", len(test_set_labels)))
    print ("{}{}".format("TestSetFeatures Size:", len(test_set_features)))
    print ("{}{}".format("TrainSetLabels Size:", len(train_set_labels)))
    print ("{}{}".format("TrainSetFeatures Size:", len(train_set_features)))
    print "\n"
    pprint.pprint(train_set_features)
    return test_set_features, test_set_labels, train_set_features, train_set_labels


def train_classifier(classifier, training_features, training_labels):
    # Set variables
    clf = classifier
    le = preprocessing.LabelEncoder()

    # Normalize labels by converting label data from type string to type int
    transformed_label_data = le.fit_transform(training_labels)
    print "Labels Transformed"

    # Save current time to variable
    start = time()

    # Train algorithm with data
    clf.fit(training_features, transformed_label_data)
    # Calculate how long it took to train algorithm
    runtime = time() - start
    print "Trained Classifier"
    print "Runtime:", round(runtime, 3), "s"

    # Return trained classifier and runtime
    return classifier, runtime


def create_label_encoder(data_set):
    data_set_labels = np.empty((0, 1), dtype=str)
    le = preprocessing.LabelEncoder()
    for i, row in data_set.iterrows():
        # Convert the row data into an array
        row_data = row.values

        # Extract the label and insert it into an array
        row_label = np.empty((0, 1), dtype=str)
        row_label = np.append(row_label, row_data[7])

        data_set_labels = np.append(data_set_labels, [row_label])
    # Fit label encoder so that it can turn labels back into text
    le.fit(data_set_labels)
    return le


def test_classifier(classifier, testing_features, testing_labels):
    # Set Variables
    clf = classifier
    le = preprocessing.LabelEncoder()
    le.fit(testing_labels)

    # Use the classifier to predict labels for the test data
    predicted_label_data = clf.predict(testing_features)

    # Convert the predicted labels back into text so they can be compared
    predicted_label_text = le.inverse_transform(predicted_label_data)
    print "Classifier Predictions:", predicted_label_text, ""

    # Calculate the accuracy of the algorithm by comparing the predicted labels to the actual labels. testing_labels are actually in or out
    accuracy = accuracy_score(predicted_label_text, testing_labels)
    print "Classifier Accuracy Score", accuracy, "\n"

    # Return accuracy
    return accuracy


def test_classifiers(amount_of_tests, data):
    # Create counter variable for loop
    counter = 1

    # Create classifiers
    clf_svm = svm.SVC()
    clf_sgd = SGDClassifier()
    clf_naive_bayes = GaussianNB()
    clf_decision_tree = tree.DecisionTreeClassifier()

    #record all runtimes and accuracies for each test so you can find the average
    svm_runtime_list = list()
    svm_accuracy_list = list()

    sgd_runtime_list = list()
    sgd_accuracy_list = list()

    naive_bayes_runtime_list = list()
    naive_bayes_accuracy_list = list()

    dec_tree_runtime_list = list()
    dec_tree_accuracy_list = list()

    while counter <= amount_of_tests:
        print "Test Number ", counter
        test_features, test_labels, train_features, train_labels = create_test_set(data, .10)

        # Train the algorithm
        print "Testing the Support Vector Machine Classifier"
        clf_svm, session_runtime_svm = train_classifier(classifier=clf_svm,
                                                        training_features=train_features,
                                                        training_labels=train_labels)

        # Test the algorithm using the test data and find the accuracy
        session_accuracy_svm = test_classifier(classifier=clf_svm,
                                               testing_features=test_features,
                                               testing_labels=test_labels)

        svm_runtime_list.append(session_runtime_svm)
        svm_accuracy_list.append(session_accuracy_svm)

        print "Testing Stochastic Gradient Decent Classifier"
        clf_sgd, session_runtime_sgd = train_classifier(classifier=clf_sgd,
                                                        training_features=train_features,
                                                        training_labels=train_labels)
        session_accuracy_sgd = test_classifier(classifier=clf_sgd,
                                               testing_features=test_features,
                                               testing_labels=test_labels)

        sgd_runtime_list.append(session_runtime_sgd)
        sgd_accuracy_list.append(session_accuracy_sgd)

        print "Testing Naive Bayes Classifier"
        clf_naive_bayes, session_runtime_naive_bayes = train_classifier(classifier=clf_naive_bayes,
                                                                        training_features=train_features,
                                                                        training_labels=train_labels)
        session_accuracy_naive_bayes = test_classifier(classifier=clf_naive_bayes,
                                                       testing_features=test_features,
                                                       testing_labels=test_labels)
        naive_bayes_runtime_list.append(session_runtime_naive_bayes)
        naive_bayes_accuracy_list.append(session_accuracy_naive_bayes)

        print "Testing Decision Tree Classifier"
        clf_decision_tree, session_runtime_dec_tree = train_classifier(classifier=clf_decision_tree,
                                                                       training_features=train_features,
                                                                       training_labels=train_labels)
        session_accuracy_dec_tree = test_classifier(classifier=clf_decision_tree,
                                                    testing_features=test_features,
                                                    testing_labels=test_labels)
        dec_tree_runtime_list.append(session_runtime_naive_bayes)
        dec_tree_accuracy_list.append(session_accuracy_naive_bayes)


        counter += 1

    print "\nSupport Vector Machine Classifier Results"
    average_accuracy_svm = float(sum(svm_accuracy_list))/len(svm_accuracy_list)
    print "Average Accuracy", average_accuracy_svm
    print "Average Runtime", float(sum(svm_runtime_list))/len(svm_runtime_list)

    print "\nStochastic Gradient Decent Classifier Results"
    average_accuracy_sgd = float(sum(sgd_accuracy_list))/len(sgd_accuracy_list)
    print "Average Accuracy", average_accuracy_sgd
    print "Average Runtime", float(sum(sgd_runtime_list))/len(sgd_runtime_list)

    print "\nNaive Bayes Classifier Results"
    average_accuracy_naive_bayes = float(sum(naive_bayes_accuracy_list))/len(naive_bayes_accuracy_list)
    print "Average Accuracy", average_accuracy_naive_bayes
    print "Average Runtime", float(sum(naive_bayes_runtime_list))/len(naive_bayes_runtime_list)

    print "\nDecision Tree Classifier Results"
    average_accuracy_dec_tree = float(sum(dec_tree_accuracy_list))/len(dec_tree_accuracy_list)
    print "Average Accuracy", average_accuracy_dec_tree
    print "Average Runtime", float(sum(dec_tree_runtime_list))/len(dec_tree_runtime_list)
    print "\n"

    top_classifier_accuracy = average_accuracy_svm
    most_accurate_classifier = clf_svm

    if average_accuracy_sgd > top_classifier_accuracy:
        top_classifier_accuracy = average_accuracy_sgd
        most_accurate_classifier = clf_sgd

    if average_accuracy_naive_bayes > top_classifier_accuracy:
        top_classifier_accuracy = average_accuracy_naive_bayes
        most_accurate_classifier = clf_naive_bayes

    if average_accuracy_dec_tree > top_classifier_accuracy:
        top_classifier_accuracy = average_accuracy_dec_tree
        most_accurate_classifier = clf_decision_tree

    return most_accurate_classifier

def parse_track_link(track_link):
    is_link_valid = False
    track_id = ""

    # Create a substring by removing the first 25 characters in the string
    link_info = track_link[25:]
    print "Link info:", link_info

    # If the link of a track link
    if link_info[:5] == "track":
        is_link_valid = True
        print "Link belongs to a track"
        # Save the track id to the track id variable
        track_id = link_info[6:]

        print "Track ID:", track_id
    else:
        print "Link does not belong to a track"

    return is_link_valid, track_id


def get_username():
    if len(sys.argv) > 1:
        # Ask the user for their username
        return sys.argv[1]
    else:
        # If username cannot be found then throw an error
        print "Usage: %s username" % (sys.argv[0],)
        sys.exit()


def create_token_for_scope(username, scope):
    # Get token for user
    token = util.prompt_for_user_token(username, scope)

    # If given valid username token
    if token:
        print ("Token Successfully Generated")
        return token
    else:
        print "Can't get token for", username


def predict_track_using_data(track_link, data):
    label_encoder = create_label_encoder(data)
    most_accurate_classifier = test_classifiers(amount_of_tests=5, data=data)

    is_track_link_valid, track_id = parse_track_link(track_link)

    scope = 'user-read-private user-read-email playlist-modify-private playlist-modify-public'

    # Create username and Token objects
    authorization_username = get_username()
    token = create_token_for_scope(scope=scope, username=authorization_username)

    # Create a Spotipy object
    sp = spotipy.Spotify(auth=token)

    track_data = sp.track(track_id)
    print "Track Name:", track_data['name']
    print "Artist:", track_data['artists'][0]['name']
    print "Album:", track_data['album']['name']
    track_audio_features = sp.audio_features([track_id])[0]

    track_features = np.empty((0, 7), dtype=float)
    track_features = np.append(track_features, [track_audio_features['acousticness']])
    track_features = np.append(track_features, [track_audio_features['danceability']])
    track_features = np.append(track_features, [track_audio_features['energy']])
    track_features = np.append(track_features, [track_audio_features['instrumentalness']])
    track_features = np.append(track_features, [track_audio_features['liveness']])
    track_features = np.append(track_features, [track_audio_features['speechiness']])
    track_features = np.append(track_features, [track_audio_features['valence']])
    track_features = np.reshape(track_features, (1, -1))

    prediction = most_accurate_classifier.predict(track_features)

    prediction = label_encoder.inverse_transform(prediction)
    print prediction
    if(prediction=="In"):
        print "Based on our predictions, this song does belong in the playlist."
    if (prediction == "Out"):
        print "Based on our predictions, this song does NOT belong in the playlist."

"""
Main Method
"""
data = pd.read_csv(filepath_or_buffer='data.csv', sep=' ')
print "Please copy and paste the link to the track you would like to predict:"
track_link = raw_input()

#below: tells us if this song belongs in the playlist based on our predictions.
#test set needed to prevent program from being over-trained. In writeup need info about why we set up test/training sets.
predict_track_using_data(track_link=track_link, data=data)
#take out method createTestSet from predict track usng data and move to beginning before asking for tracks