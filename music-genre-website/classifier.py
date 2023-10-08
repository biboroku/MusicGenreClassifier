#import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from datetime import datetime
import numpy as np
import seaborn as sns
#import plotly.express as px
import math, os
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing

def getSpotifyPlaylist(client_id, client_secret, url, genre):
    '''
    Main function that downloads a Spotify playlist as an Excel file with key features for each song via API calls
    '''
    # Set up API query with credentials
    client_credentials_manager = SpotifyClientCredentials(client_id = client_id, client_secret = client_secret)
    sp_query = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    sp = spotipy.Spotify(
                client_credentials_manager=SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret))

    # Download the playlist as an Excel file
    downloadPlaylistAsExcel(url, genre, sp)

def downloadPlaylistAsExcel(url, genre, sp):
    '''
    Helper function that retrieves all songs from an identified playlist, and takes the ID, song, album, artist 
    and puts this all into a DataFrame
    '''
   # SONG NAMES

    offset=0
    name = []

    while True:
        response = sp.playlist_tracks(url,
                                      offset=offset,
                                      fields=['items.track.name,total'])

        name.append(response["items"])
        offset = offset + len(response['items'])
        
        if len(response['items']) == 0:
            break

    name_list = [b["track"]["name"] for a in name for b in a]
    len(name_list)
    
    

    # ALBUM

    offset=0
    album = []

    while True:
        response = sp.playlist_tracks(url,
                                      offset=offset,
                                      fields=['items.track.album.name,total'])

        album.append(response["items"])
        offset = offset + len(response['items'])
        
        if len(response['items']) == 0:
            break

    album_list = [b["track"]["album"]["name"] for a in album for b in a]

    
    
   # ARTIST

    offset=0
    artist = []

    while True:
        response = sp.playlist_tracks(url,
                                      offset=offset,
                                      fields=['items.track.album.artists.name,total'])

        artist.append(response["items"])
        offset = offset + len(response['items'])
        
        if len(response['items']) == 0:
            break

    artist_list = [b["track"]["album"]["artists"][0]["name"] for a in artist for b in a]

    
    # ID
    
    offset = 0
    identifier = []

    while True:
        response = sp.playlist_tracks(url,
                                      offset=offset,
                                      fields=['items.track.id,total'])

        identifier.append(response["items"])
        offset = offset + len(response['items'])
        
        if len(response['items']) == 0:
            break

    identifier_list= [b["track"]["id"] for a in identifier for b in a]
    len(identifier_list)

    #Get audio features
    features = [sp.audio_features(identifier) for identifier in identifier_list]
    
    # Get each invidividual feature
    danceability = [(b["danceability"]) for a in features for b in a]    
    mode = [(b["mode"]) for a in features for b in a]
    energy = [(b["energy"]) for a in features for b in a]
    key = [(b["key"]) for a in features for b in a]        
    loudness = [(b["loudness"]) for a in features for b in a]       
    speechiness = [(b["speechiness"]) for a in features for b in a]
    acousticness = [(b["acousticness"]) for a in features for b in a]        
    instrumentalness = [(b["instrumentalness"]) for a in features for b in a] 
    liveness = [(b["liveness"]) for a in features for b in a]
    valence = [(b["valence"]) for a in features for b in a]        
    tempo = [(b["tempo"]) for a in features for b in a] 
    duration_ms = [(b["duration_ms"]) for a in features for b in a] 
    identifier_ = [(b["id"]) for a in features for b in a] 
    
    ## DataFrame (saved with current time)

    df = pd.DataFrame({"Song name": name_list, "Artist": artist_list, "Album": album_list, "ID": identifier_list})
    df_2 = pd.DataFrame({"Danceability":danceability,
                         "Mode":mode,
                         "Energy":energy,
                         "Key":key,
                         "Loudness":loudness,
                         "Speechiness":speechiness,
                         "Acousticness":acousticness,
                         "Instrumentalness":instrumentalness,
                         "Liveness":liveness,
                         "Valence":valence,
                         "Tempo":tempo,
                         "Duration (ms)": duration_ms,
                         "ID_CHECK":identifier_
                               })

    df_combined = df_2.join(df)
    
    df_combined.to_excel(os.getcwd()+"/" + genre + ".xlsx")

    return df_combined.tail()

def trainKnnClassifier(dataFile1, dataFile2, n_neighbors):
    '''
    Main function that trains and tests a kNN classifier for two classes. 
    Returns the trained classifier.
    '''
    labelled_data1 = pd.read_excel(dataFile1, index_col = 0, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12])
    labelled_data2 = pd.read_excel(dataFile2, index_col = 0, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12])
    labelled_data1["Class"] = 0
    labelled_data2["Class"] = 1
    
    # concatenate and randomize the two data sets
    data = [labelled_data1, labelled_data2]
    full_data = pd.concat(data)
    full_data["Key"] = (full_data["Key"] / full_data["Key"].max())
    full_data["Loudness"] = (full_data["Loudness"] / full_data["Loudness"].min())
    full_data["Tempo"] = (full_data["Tempo"] / full_data["Tempo"].max())
    full_data_random = full_data.sample(frac=1)
    
    # Get training data with train-test split equals 70-30
    rows = full_data_random.shape[0]

    X_train = full_data_random.iloc[0:int(rows*0.7),:]
    Y_train = X_train["Class"].values
    X_train = X_train.drop("Class", axis=1)

    # Get test data with train-test split equals 70-30
    X_test = full_data_random.iloc[int(rows*0.7):,:]

    # Randomize the data
    X_test = X_test.sample(frac=1)
    Y_test = X_test["Class"].values
    X_test = X_test.drop("Class", axis=1)
    
    # Fit kNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, Y_train)
    # cross_validate default to 5-fold CV
    cv_scores = cross_val_score(knn, X_train, Y_train, cv=5)
    cv_scores_mean = np.mean(cv_scores)
    
    # Test prediction
    predictions = knn.predict(X_test)
    accuracy_score = knn.score(X_test, Y_test)
    print("Accuracy score = ""{:.2f}".format(accuracy_score))
    confusion_matrix = metrics.confusion_matrix(Y_test, predictions)
    print("Confusion Matrix: \n", confusion_matrix)
    
    return knn

def trainDecisionTreeClassifier(dataFile1, dataFile2):
    '''
    Main function that trains and tests a kNN classifier for two classes. 
    Returns the trained classifier.
    '''
    labelled_data1 = pd.read_excel(dataFile1, index_col = 0, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12])
    labelled_data2 = pd.read_excel(dataFile2, index_col = 0, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12])
    labelled_data1["Class"] = 0
    labelled_data2["Class"] = 1
    
    # concatenate and randomize the two data sets
    data = [labelled_data1, labelled_data2]
    full_data = pd.concat(data)
    full_data["Key"] = (full_data["Key"] / full_data["Key"].max())
    full_data["Loudness"] = (full_data["Loudness"] / full_data["Loudness"].min())
    full_data["Tempo"] = (full_data["Tempo"] / full_data["Tempo"].max())
    full_data_random = full_data.sample(frac=1)
    
    # Get training data with train-test split equals 70-30
    rows = full_data_random.shape[0]

    X_train = full_data_random.iloc[0:int(rows*0.7),:]
    Y_train = X_train["Class"].values
    X_train = X_train.drop("Class", axis=1)

    # Get test data with train-test split equals 70-30
    X_test = full_data_random.iloc[int(rows*0.7):,:]

    # Randomize the data
    X_test = X_test.sample(frac=1)
    Y_test = X_test["Class"].values
    X_test = X_test.drop("Class", axis=1)
    
    # Create Decision Tree classifier object
    tree = DecisionTreeClassifier(criterion="gini", splitter="best")

    # Train Decision Tree classifier
    tree = tree.fit(X_train, Y_train)

    # Predict the response for test dataset
    Y_pred = tree.predict(X_test)
    print("Accuracy score = ""{:.2f}".format(metrics.accuracy_score(Y_test, Y_pred)))
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix: \n", confusion_matrix)
    
    return tree

def predictWithKnn(client_id, client_secret,playlist_uri, genre="unknown"):
    '''
    Function returns a prediction for the genre of a song given its Spotify url
    '''
    getSpotifyPlaylist(client_id, client_secret, playlist_uri, genre)
    input_data = pd.read_excel(os.getcwd()+"/{}.xlsx".format(genre),index_col = 0,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12])
    predictions = knn.predict(input_data)
    return predictions

def predictWithTree(client_id, client_secret,playlist_uri, genre="unknown"):
    '''
    Function returns a prediction for the genre of a song given its Spotify url
    '''
    getSpotifyPlaylist(client_id, client_secret, playlist_uri, genre)
    input_data = pd.read_excel(os.getcwd()+"/{}.xlsx".format(genre),index_col = 0,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12])
    predictions = tree.predict(input_data)
    return predictions