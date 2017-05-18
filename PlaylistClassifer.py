"""
This class takes two text files of JSON Formatted data for a Spotify playlist and 
trains a classification algorithm based on the data in the "Training Set" text file. 

The algorithm is then tested with the "Test Set" text file and the accuracy score is output. 
The algorithm can then be used give the fit of a link to any song on Spotify
"""

import pandas as pd

def createTestSet(dataSet, testSetProportion):
    testSetSize = int(len(dataSet)*testSetProportion)
    print ("{}{}".format("DataSet Size:", len(dataSet)))
    print ("{}{}".format("Test set Size:", testSetSize))


data = pd.read_csv(filepath_or_buffer='data.csv', sep=' ')
print (data)
createTestSet(data, .10)
