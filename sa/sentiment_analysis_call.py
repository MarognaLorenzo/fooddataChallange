from nltk.corpus import twitter_samples # importing samples
import numpy as np
from scipy.signal import freqs

import sentiment_analysis as sentiment_analysis
import pandas as pd
import os

# BUCKET_URL="https://d4gc2024.s3.eu-central-1.amazonaws.com/data"
BASE_URL = "../data"
filenames=[
    "foodsavingleuven/places.csv",
    "foodsavingleuven/members.csv"	,
    "foodsavingleuven/activities.csv",
    "foodsavingleuven/activities_participants.csv",
    "foodsavingleuven/feedback.csv"
]

def train():
    positive_sample = twitter_samples.strings('positive_tweets.json')
    negative_sample = twitter_samples.strings('negative_tweets.json')

    # How many samples do we have ?
    # print(len(positive_sample)) # 5000
    # print(len(negative_sample)) # 5000

    # 1) Training and Testing Arrays setup
    train_x = positive_sample[:4000] + negative_sample[:4000]
    test_x = positive_sample[4000:] + negative_sample[4000:]

    # Combine positive and negative labels
    train_y = np.append(np.ones((len(positive_sample[:4000]), 1)), np.zeros((len(negative_sample[:4000]), 1)), axis=0)
    test_y = np.append(np.ones((len(positive_sample[4000:]), 1)), np.zeros((len(negative_sample[4000:]), 1)), axis=0)

    # 2) Train model using the training dataset and test it on the test dataset
    freqs = sentiment_analysis.build_freqs(positive_sample, negative_sample)
    X = np.zeros((len(train_x), 3))
    for i in range(len(train_x)):
        X[i, :] = sentiment_analysis.features_extraction(train_x[i], freqs)

    theta = sentiment_analysis.learn(X, train_y, 1e-9, 1000)

    return sentiment_analysis, freqs, theta

