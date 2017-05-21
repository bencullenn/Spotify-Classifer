"""
This class takes a link to a Spotify playlist and converts that playlist to a training set 
which can be used to train machine learning algorithms in the form of a text file
"""

import sys
import spotipy
import spotipy.util as util
import pprint
import csv

'''
Functions
'''
def getUsername():
    if len(sys.argv) > 1:
        # Ask the user for their username
        return sys.argv[1]
    else:
        # If username cannot be found then throw an error
        print "Usage: %s username" % (sys.argv[0],)
        sys.exit()


def createTokenForScope(username,scope):
    # Get token for user
    token = util.prompt_for_user_token(username, scope)

    # If given valid username token
    if token:
        print ("Token Successfully Generated")
        return token
    else:
        print "Can't get token for", username


def getPlaylistForID(playlistID):
    # Retrieve the playlist data in JSON format
    return sp.user_playlist(username, playlistID)


def getTracksFromPlaylist(playlistID):
    # Retrieve the playlist data in JSON format
    playlist = getPlaylistForID(playlistID)

    # Create two list objects, one to store all of JSON data for the tracks in the playlist
    # and another to store the uri link data for audio feature processing
    tracks = []

    # Get the total number of tracks in the playlist and create a variable to manage the offset value
    playlistNumOfTracks = playlist['tracks']['total']
    offset = 0

    # While the tracks list contains less than the number of tracks
    # in the playlist add chunks of 100 songs at a time to the tracks playlist
    while len(tracks) < playlistNumOfTracks:
        linkListSubset = sp.user_playlist_tracks(username, playlistID, fields='items', offset=offset)['items']
        for track in linkListSubset:
            tracks.append(track)
        offset += 1
        print("{}{}".format("Length of playlist:", playlistNumOfTracks))
        print("{}{}".format("Current amount of tracks retrieved:", len(tracks)))
        print("{}{}".format("Offset Value:", offset))
    return tracks


def getAudioFeaturesForPlaylistID(playlistID):
    tracks = getTracksFromPlaylist(playlistID)
    links = []

    for track in tracks:
        links.append(track['track']['uri'])

    # Create variables to store the beginning and end indies to enable getting subsets of results
    startIndex = 0
    endIndex = 100
    audioFeatures = []

    # While the amount of audio feature data is less than the amount of links
    while len(audioFeatures) < len(links):

        # Adjust the end index to be at max the size of the list of links to avoid an index out of bounds error
        if len(links) < endIndex:
            endIndex = len(links)
            print("{}{}".format("End index updated to:", endIndex))

        # Create a temporary list
        linkListSubset = []

        # For the range of indexes remove the links from the link list and add them to the temp list
        for index in range(startIndex, endIndex):
            linkListSubset.append(links[index])

        # Retrieve the audio features for the subset of the links
        audioFeaturesRawData = sp.audio_features(linkListSubset)

        # Add the raw data for each track to the audio features list
        for item in audioFeaturesRawData:
            audioFeatures.append(item)

        # Change the indices to include the next 100 songs
        startIndex = endIndex
        endIndex += 100

        # Print some information to know what's happening
        print("{}{}".format("Amount of links and audio features are the same:", len(audioFeatures) == len(links)))
        print("{}{}".format("Amount of Audio Features:", len(audioFeatures)))
        print("{}{}".format("Amount of links:", len(links)))
        print("{}{}".format("Start Index:", startIndex))
        print("{}{}".format("End Index:", endIndex))
        print "\n"
    return audioFeatures


def displayPlaylist(playlistID):
    tracks = getTracksFromPlaylist(playlistID)
    playlist = getPlaylistForID(playlistID)

    # Print out playlist name
    print playlist['name']
    print"___________________________\n"

    # Print out the name, artist, album, and link for each song in the playlist and add uri data to links array
    for track in tracks:
        print track['track']['name']
        print track['track']['artists'][0]['name']
        print track['track']['album']['name']
        print track['track']['external_urls']['spotify']
        print "\n"


def writeAudioFeaturesToCSVFile(positiveAudioFeatures,negativeAudioFeatures):
    # Create a new csv file
    with open('data.csv', 'wb') as csvfile:
        # Create a writer for the csv file
        dataWriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # Write a header for the CSV
        dataWriter.writerow(['Acoustic',
                             'Dance',
                             'Energy',
                             'Instrumental',
                             'Live',
                             'Speech',
                             'Valence',
                             'Label'])

        # For each track write certain attributes to the csv file
        for featureSet in positiveAudioFeatures:
            dataWriter.writerow([featureSet['acousticness'],
                                 featureSet['danceability'],
                                 featureSet['energy'],
                                 featureSet['instrumentalness'],
                                 featureSet['liveness'],
                                 featureSet['speechiness'],
                                 featureSet['valence'], 1])

        for featureSet in negativeAudioFeatures:
            dataWriter.writerow([featureSet['acousticness'],
                                 featureSet['danceability'],
                                 featureSet['energy'],
                                 featureSet['instrumentalness'],
                                 featureSet['liveness'],
                                 featureSet['speechiness'],
                                 featureSet['valence'], 0])

        print("Data successfully written to csv file")

"""
Variable Declarations
"""
# Define the scope of what you would like to access from the user
scope = 'user-read-private user-read-email'

# Get the ID's for the playlists of your postive and negative examples
negativeExamplesPlaylistID = "0Y6fI3iYSYSfZY1B7X3tvU"
positiveExamplesPlaylistID = "6KNC0KnNsw7hJUGwlr1hCO"

# Create username and Token objects
username = getUsername()
token = createTokenForScope(scope=scope, username=username)

# Create a Spotipy object
sp = spotipy.Spotify(auth=token)

positiveExamplesData = getAudioFeaturesForPlaylistID(positiveExamplesPlaylistID)
negativeExamplesData = getAudioFeaturesForPlaylistID(negativeExamplesPlaylistID)

writeAudioFeaturesToCSVFile(positiveAudioFeatures=positiveExamplesData, negativeAudioFeatures=negativeExamplesData)