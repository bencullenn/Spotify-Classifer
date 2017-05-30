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


def get_playlist_for_id(playlist_id):
    # Retrieve the playlist data in JSON format
    return sp.user_playlist(authorization_username, playlist_id)


def get_tracks_from_playlist(playlist_id):
    # Retrieve the playlist data in JSON format
    playlist = get_playlist_for_id(playlist_id)

    # Create two list objects, one to store all of JSON data for the tracks in the playlist
    # and another to store the uri link data for audio feature processing
    tracks = []

    # Get the total number of tracks in the playlist and create a variable to manage the offset value
    playlist_num_of_tracks = playlist['tracks']['total']
    offset_value = 0

    # While the tracks list contains less than the number of tracks
    # in the playlist add chunks of 100 songs at a time to the tracks playlist
    while len(tracks) < playlist_num_of_tracks:
        link_list_subset = sp.user_playlist_tracks(authorization_username, playlist_id, fields='items', offset=offset_value)['items']
        for track in link_list_subset:
            tracks.append(track)
        offset_value += 1
        print("{}{}".format("Length of playlist:", playlist_num_of_tracks))
        print("{}{}".format("Current amount of tracks retrieved:", len(tracks)))
        print("{}{}".format("Offset Value:", offset_value))
    return tracks


def get_audio_features_for_playlist_id(playlist_id):
    tracks = get_tracks_from_playlist(playlist_id)
    links = []

    for track in tracks:
        links.append(track['track']['uri'])

    # Create variables to store the beginning and end indies to enable getting subsets of results
    start_index = 0
    end_index = 100
    audio_features = []

    # While the amount of audio feature data is less than the amount of links
    while len(audio_features) < len(links):

        # Adjust the end index to be at max the size of the list of links to avoid an index out of bounds error
        if len(links) < end_index:
            end_index = len(links)
            print("{}{}".format("End index updated to:", end_index))

        # Create a temporary list
        link_list_subset = []

        # For the range of indexes remove the links from the link list and add them to the temp list
        for index in range(start_index, end_index):
            link_list_subset.append(links[index])

        # Retrieve the audio features for the subset of the links
        audio_features_raw_data = sp.audio_features(link_list_subset)

        # Add the raw data for each track to the audio features list
        for item in audio_features_raw_data:
            audio_features.append(item)

        # Change the indices to include the next 100 songs
        start_index = end_index
        end_index += 100

        # Print some information to know what's happening
        print("{}{}".format("Amount of links and audio features are the same:", len(audio_features) == len(links)))
        print("{}{}".format("Amount of Audio Features:", len(audio_features)))
        print("{}{}".format("Amount of links:", len(links)))
        print("{}{}".format("Start Index:", start_index))
        print("{}{}".format("End Index:", end_index))
        print "\n"
    return audio_features


def display_playlist(playlist_id):
    tracks = get_tracks_from_playlist(playlist_id)
    playlist = get_playlist_for_id(playlist_id)

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


def write_audio_features_to_csv_file(positive_audio_features, negative_audio_features):
    # Create a new csv file
    with open('data.csv', 'wb') as csvfile:
        # Create a writer for the csv file
        data_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # Write a header for the CSV
        data_writer.writerow(['Acoustic',
                              'Dance',
                              'Energy',
                              'Instrumental',
                              'Live',
                              'Speech',
                              'Valence',
                              'Label'])

        # Create variables for tracking indices
        negative_example_index = 0
        positive_example_index = 0

        # Get size of datasets
        postive_example_size = len(positive_audio_features)
        negative_examples_size = len(negative_audio_features)

        # While there are more positive data examples
        while positive_example_index < postive_example_size:

            data_writer.writerow([positive_audio_features[positive_example_index]['acousticness'],
                                 positive_audio_features[positive_example_index]['danceability'],
                                 positive_audio_features[positive_example_index]['energy'],
                                 positive_audio_features[positive_example_index]['instrumentalness'],
                                 positive_audio_features[positive_example_index]['liveness'],
                                 positive_audio_features[positive_example_index]['speechiness'],
                                 positive_audio_features[positive_example_index]['valence'], "In"])
            positive_example_index += 1

        # While there are more negative data examples
        while negative_example_index < negative_examples_size:
            # Write certain data to the csv file
            data_writer.writerow([negative_audio_features[negative_example_index]['acousticness'],
                                 negative_audio_features[negative_example_index]['danceability'],
                                 negative_audio_features[negative_example_index]['energy'],
                                 negative_audio_features[negative_example_index]['instrumentalness'],
                                 negative_audio_features[negative_example_index]['liveness'],
                                 negative_audio_features[negative_example_index]['speechiness'],
                                 negative_audio_features[negative_example_index]['valence'], "Out"])

            negative_example_index += 1

        print("Data successfully written to csv file")


def parse_playlist_link(playlist_link):
    is_playlist = False
    creator_username = ""
    playlist_id = ""

    link_info = playlist_link[25:]

    print "Link info:", link_info

    if link_info[:4] == "user":
        is_playlist = True
        print "Link belongs to a playlist"
        link_info = link_info[5:]
    else:
        print "Link does not belong to a playlist"

    if is_playlist:
            for character in link_info:
                if character is not "/":
                    creator_username += character
                else:
                    break
            print "Username:", creator_username

            link_info = link_info[len(creator_username)+10:]

            print "Link info without username:", link_info
            playlist_id = link_info
            print "\n"
    return is_playlist, creator_username, playlist_id


"""
Main Code
"""
# Define the scope of what you would like to access from the user
scope = 'user-read-private user-read-email'

# Get the ID's for the playlists of your positive and negative examples
print"Please input the link to your playlist of positive examples:"
positive_examples_playlist_link = raw_input()
print"Please input the link to your playlist of negative examples:"
negative_examples_playlist_link = raw_input()


# Create username and Token objects
authorization_username = get_username()
token = create_token_for_scope(scope=scope, username=authorization_username)

# Create a Spotipy object
sp = spotipy.Spotify(auth=token)

is_pos_playlist_link_valid, pos_playlist_username, pos_playlist_id = parse_playlist_link(positive_examples_playlist_link)
is_neg_playlist_link_valid, neg_playlist_username, neg_playlist_id = parse_playlist_link(negative_examples_playlist_link)

if is_pos_playlist_link_valid:
    if is_neg_playlist_link_valid:
        positive_examples_data = get_audio_features_for_playlist_id(pos_playlist_id)
        negative_examples_data = get_audio_features_for_playlist_id(neg_playlist_id)
        write_audio_features_to_csv_file(positive_audio_features=positive_examples_data,
                                         negative_audio_features=negative_examples_data)
    else:
        print "Negative playlist link is not valid"
else:
    print "Positive playlist link is not valid"
