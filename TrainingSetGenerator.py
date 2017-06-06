"""
This class takes two links from two Spotify playlists, one of positve examples and one of negative examples,
and converts that playlist to a training set which can be used to train machine learning algorithms
in the form of a csv file
"""

import sys
import spotipy
import spotipy.util as util
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
    # Get api_token for user
    token = util.prompt_for_user_token(username, scope)

    # If given valid username api_token
    if token:
        print "API Token Successfully Generated"
        return token
    else:
        print "Can't get token for Username:", username


def get_playlist_for_id(playlist_id):
    # Retrieve the playlist data in JSON format
    return sp.user_playlist(authorization_username, playlist_id)


def get_tracks_from_playlist(playlist_id):
    # Retrieve the playlist data in JSON format
    playlist = get_playlist_for_id(playlist_id)

    # Create two list objects, one to store all of JSON data for the tracks in the playlist
    # and another to store the uri link data for audio feature processing
    tracks = []

    # Get the total number of tracks in the playlist and create a variables to manage the offset and limit values
    playlist_num_of_tracks = playlist['tracks']['total']
    offset_value = 0
    limit = 100

    # While the tracks list contains less than the number of tracks
    # in the playlist add chunks of 100 songs at a time to the tracks playlist
    while len(tracks) < playlist_num_of_tracks:
        playlist_tracks_subset = sp.user_playlist_tracks(authorization_username,
                                                         playlist_id,
                                                         fields='items',
                                                         offset=offset_value,
                                                         limit=limit)['items']

        # Add the retrieved tracks to the list of all the tracks
        for track in playlist_tracks_subset:
            tracks.append(track)

        # Update the offset value
        offset_value += 1

        # Check if the playlist contains less than 100 more songs and adjust the limit value accordingly
        if (offset_value+1)*100 > playlist_num_of_tracks:
            difference = (offset_value+1)*100 - playlist_num_of_tracks
            limit = 100 - difference

        print "Retrieved ", len(tracks), " of ", playlist_num_of_tracks, " tracks"

    print"\n"
    return tracks


def get_audio_features_for_playlist_id(playlist_id):
    tracks = get_tracks_from_playlist(playlist_id)
    links = []

    # link is not actual url, but string of numbers that corresponds to track.
    for track in tracks:
        links.append(track['track']['uri'])

    # Create variables to store the beginning and end indices to enable getting subsets of results
    start_index = 0
    end_index = 100
    audio_features = []

    # While the amount of audio feature data is less than the amount of links
    while len(audio_features) < len(links):

        # Adjust the end index to be at max the size of the list of links to avoid an index out of bounds error
        if len(links) < end_index:
            end_index = len(links)

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
        print"Retrieved audio features for ", len(audio_features), " of ", len(links), " tracks"

    print "\n"
    return audio_features


def display_playlist(playlist_id):
    tracks = get_tracks_from_playlist(playlist_id)
    playlist = get_playlist_for_id(playlist_id)

    # Print out playlist name
    print playlist['name']
    print"___________________________\n"

    # Print out the name, artist, album, and link for each song in the playlist
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

        # Get size of data sets
        positive_example_size = len(positive_audio_features)
        negative_examples_size = len(negative_audio_features)

        # While there are more positive data examples
        while positive_example_index < positive_example_size:

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

    if link_info[:4] == "user":
        is_playlist = True
        link_info = link_info[5:]
        print "Link belongs to playlist \"", get_playlist_for_id(link_info)['name'], "\""
    else:
        print "Link does not belong to a playlist"

    if is_playlist:
            for character in link_info:
                if character is not "/":
                    creator_username += character
                else:
                    break

            link_info = link_info[len(creator_username)+10:]

            playlist_id = link_info
    return is_playlist, creator_username, playlist_id


"""
Main Method
"""
# Define the scope of information you would like to access from the user
account_access_scope = 'user-read-private user-read-email'

# Get the links for the playlists of your positive and negative examples
print"Please input the link to your playlist of positive examples:"
positive_examples_playlist_link = raw_input()
print"Please input the link to your playlist of negative examples:"
negative_examples_playlist_link = raw_input()


# Create username and API Token objects
authorization_username = get_username()
api_token = create_token_for_scope(scope=account_access_scope, username=authorization_username)

# Create a Spotipy object
sp = spotipy.Spotify(auth=api_token)

print "\n"
print "Parsing positive examples playlist link"
is_pos_playlist_link_valid, pos_playlist_username, pos_playlist_id = parse_playlist_link(positive_examples_playlist_link)

print "Parsing negative examples playlist link"
is_neg_playlist_link_valid, neg_playlist_username, neg_playlist_id = parse_playlist_link(negative_examples_playlist_link)
print "\n"

if is_pos_playlist_link_valid:
    if is_neg_playlist_link_valid:
        print "Fetching data for positive examples playlist"
        positive_examples_data = get_audio_features_for_playlist_id(pos_playlist_id)

        print "Fetching data for negative examples playlist"
        negative_examples_data = get_audio_features_for_playlist_id(neg_playlist_id)

        write_audio_features_to_csv_file(positive_audio_features=positive_examples_data,
                                         negative_audio_features=negative_examples_data)
    else:
        print "Negative Examples playlist link is not valid"
else:
    print "Positive Examples playlist link is not valid"
