class Song(object):

    def __init__(self, acousticness, danceability, energy, instrumentalness, liveness, speechiness, valence):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.acousticness = acousticness
        self.danceability = danceability
        self.energy = energy
        self.instrumentalness = instrumentalness
        self.liveness = liveness
        self.speechiness = speechiness
        self.valence = valence
