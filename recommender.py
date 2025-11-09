from EmotionClassifier import EmotionClassifier
import spotipy
from help import emotion_to_genre,artist_patterns,emotion_keywords
import re
from spotipy.oauth2 import SpotifyClientCredentials
import random

class MoodRecommender:
    def __init__(self,client_id,client_secret):
        self.classifier=EmotionClassifier()
        self.sp=spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ))

    def extract_artist_name(self,text):
        for pattern in artist_patterns:
            match=re.search(pattern,text,re.IGNORECASE)
            if match:
                artist=match.group(1).strip()
                artist=' '.join(word.capitalize() for word in artist.split())

                #Filter out common false +ve s:
                common_words=['the', 'a', 'an', 'i', 'my', 'me', 'you', 'it', 'is', 'are', 'and', 'or', 'but']
                if len(artist) > 2 and artist.lower() not in common_words:
                    return artist

        return None

    def result(self,results,emotion_labels,genre,artist_name=None):
        track = random.choice(results['tracks']['items'])
        name = track['name']
        artist = track['artists'][0]['name']
        url = track['external_urls']['spotify']

        print(f"Recommended Song: {name} by {artist}")
        print(f"{url}")

        result_dict={
            "emotion": emotion_labels,
            "genre": genre,
            "song_name": name,
            "artist": artist,
            "spotify_url": url
        }

        if artist_name:
            result_dict["artist_requested"] = artist_name

        return result_dict

    def recommend_by_mood_keywords(self,artist_name,emotion):
        keywords=emotion_keywords.get(emotion,[])
        if not keywords:
            return None

        try:
            results=self.sp.search(q=f'artist:{artist_name}',type='track',limit=50)

            if not results['tracks']['items']:
                return None

            tracks=results['tracks']['items']
            print(f'Analyzing {len(tracks)} tracks for mood keywords...')

            #score based on keyword matches
            scored_tracks=[]
            for track in tracks:
                score=0
                track_name=track['name'].lower()
                album_name=track['album']['name'].lower() if track.get('album') else ''

                #checking for keyword matches
                for keyword in keywords:
                    if keyword in track_name:
                        score+=3
                    if keyword in album_name:
                        score+=1

                if score>0:
                    scored_tracks.append((track,score))

            if not scored_tracks:
                print(f"No tracks found matching mood keywords for {emotion}")
                return None

            #sort by score
            scored_tracks.sort(key=lambda x:x[1],reverse=True)
            print(f"Found {len(scored_tracks)} tracks with keyword matches")

            if scored_tracks:
                print(f"Top matches: ")
                for i,(t,s) in enumerate(scored_tracks[:3],1):
                    print(f"{i}. '{t['name']}' (score:{s})")

            top_matches=scored_tracks[:10]
            track,score=random.choice(top_matches)
            name=track['name']
            artist=track['artists'][0]['name']
            url=track['external_urls']['spotify']

            print(f"Selected: {name} by {artist} (match score: {score})")
            print(f"{url}")

            return {
                "emotion": [emotion],
                "matching_method": "mood_keywords",
                "match_score": score,
                "artist_requested": artist_name,
                "song_name": name,
                "artist": artist,
                "spotify_url": url
            }

        except Exception as e:
            print(f"Error in mood keyword matching: {e}")
            return None


    def recommend_song(self,text):

        emotions=self.classifier.predict(text)
        print(f"Detected emotions: {emotions}")

        #Extract emotions
        if isinstance(emotions,dict) and 'predictions' in emotions:
            emotion_labels=[emotion[0] for emotion in emotions['predictions']]
        else:
            emotion_labels=emotions

        print(f"Emotion labels: {emotion_labels}")

        #Find matching genres for emotion
        genres=['pop']
        for emotion in emotion_labels:
            if emotion in emotion_to_genre:
                genres=emotion_to_genre[emotion]
                break

        print(f"Available genres for emotion: {genres}")

        artist_name=self.extract_artist_name(text)

        if artist_name:
            print(f"Detected artist: {artist_name}")

            #try each genre with artist
            for genre in genres:
                print(f"Trying genre: {genre} with artist: {artist_name}")
                results=self.sp.search(q=f'artist:{artist_name} genre:{genre}',type='track',limit=50)
                if results['tracks']['items']:
                    return self.result(results,emotion_labels,genre,artist_name)

            #if no genre with artist match, try mood-based keyword filter
            print("No songs found with genre artist filter, trying mood-based keyword matching...")
            recommended_track=self.recommend_by_mood_keywords(artist_name,emotion_labels[0])

            if recommended_track:
                return recommended_track

            #if mood matching fails
            print("Mood matching failed, returning random song by artist..")
            results=self.sp.search(q=f'artist:{artist_name}',type='track',limit=50)
            if results['tracks']['items']:
                return self.result(results,emotion_labels,genres[0],artist_name)

        #if no artist mentioned, use genre-based search
        genre=random.choice(genres)
        print(f"No artist mentioned, using genre based search with genre: {genre}")
        results=self.sp.search(q=f'genre:{genre}',type='track',limit=50)

        if results['tracks']['items']:
            return self.result(results, emotion_labels, genres[0])
        else:
            print("No song found for this mood.")
            return {"emotion": emotion_labels, "genre": genre, "message": "No song found"}


