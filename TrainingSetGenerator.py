"""
This class takes a link to a Spotify playlist and converts that playlist to a training set 
which can be used to train machine learning algorithms in the form of a text file
"""

import sys
import spotipy
import spotipy.util as util
import pprint
import json

# Define the scope of what you would like to access from the user
scope = 'user-read-private user-read-email'

if len(sys.argv) > 1:
    # Ask the user for their username
    username = sys.argv[1]
else:
    # If username cannot be found then throw an error
    print "Usage: %s username" % (sys.argv[0],)
    sys.exit()

# Get token for user
token = util.prompt_for_user_token(username, scope)

if token:
    # Create a new Spotipy object with the token
    sp = spotipy.Spotify(auth=token)
    print ("Authorization Successful\n")
else:
    print "Can't get token for", username


playlist = sp.user_playlist(username, '6KNC0KnNsw7hJUGwlr1hCO')
tracks = playlist['tracks']['items']

# Print out playlist name
print playlist['name']
print"___________________________\n"

# Print out playlist JSON data
links = []

for track in tracks:
    print track['track']['name']
    print track['track']['artists'][0]['name']
    print track['track']['album']['name']
    print track['track']['external_urls']['spotify']
    links.append(track['track']['uri'])

    print "\n"

audioFeatures = sp.audio_features(links)

with open('data.txt', 'w') as outfile:
    json.dump(audioFeatures, outfile)
