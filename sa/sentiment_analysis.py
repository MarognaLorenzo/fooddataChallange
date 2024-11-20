import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import numpy as np
from collections import defaultdict


# LOGISTIC REGRESSION IMPLEMENTATION
# Sigmoid function (outputs a probability between 0 and 1)
# @param    x -> the input (it can be a scalar or an array)
# @returns  h -> the sigmoid of x
def sigmoid(x):
    h = 1 / (1 + np.exp(-x))
    return h


# Learning function
# @param   x         -> matrix of features which is (m,n+1)
# @param   y         -> corresponding labels of the input matrix x, dimensions (m,1)
# @param   alpha     -> learning rate
# @param   num_iters -> number of iterations you want to train your model
# @returns theta     -> final weight vector
def learn(x, y, alpha, num_iters):
    m = x.shape[0]
    theta = np.zeros((3, 1))
    for _ in range(0, num_iters):
        z = x @ theta
        h = sigmoid(z)
        # update the weights theta
        theta -= (alpha / m) * (x.T @ (h - y))
    return theta


# Preprocessing tweets
# clean, tokenize and stem the tweet given as argument
def process_text(tweet):
    # clean
    tweet2 = re.sub(r'^RT[\s]', '', tweet)  # remove old style retweet text "RT"
    tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2)  # remove hyperlinks
    tweet2 = re.sub(r'#', '', tweet2)  # remove # from hashtags

    # tokenize
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet2)

    # clean from stopwords and punctuation
    english_stopwords = stopwords.words('english')
    tweet_clean = []
    for word in tokens:
        if word in english_stopwords or word in string.punctuation:
            continue
        tweet_clean.append(word)

    # stem
    stemmer = PorterStemmer()
    tweet_stems = []
    for word in tweet_clean:
        tweet_stems.append(stemmer.stem(word))

    return tweet_stems


# Build frequency dictionary (related to positive/negative label)
def build_freqs(positive_sample, negative_sample):
    freqs = defaultdict(lambda: 0)
    for sample, label in [(positive_sample, 1), (negative_sample, 0)]:
        for tweet in sample:
            word_list = process_text(tweet)
            for word in word_list:
                freqs[(word, label)] += 1
    return freqs


# Extract features
def features_extraction(tweet, freqs):
    word_l = process_text(tweet)
    x = np.zeros((1, 3))  # 3 elements in the form of a 1x3 vector
    x[0, 0] = 1  # bias is set to 1
    for word in word_l:
        x[0, 1] += freqs.get((word, 1), 0)  # increment the word count for the positive label 1
        x[0, 2] += freqs.get((word, 0), 0)  # increment the word count for the negative label 0

    assert (x.shape == (1, 3))
    return x


# Predict label extracting features
# if the value predicted is >= 0.5 is considered as 1, otherwise 0
def predict(tweet, freqs, theta):
    x = features_extraction(tweet, freqs)
    y_pred = sigmoid((x @ theta))
    return 1 if y_pred >= 0.5 else 0


# Compute accuracy comparing predicted values with the original ones
def compute_accuracy(test_x, test_y, freq, theta):
    y_hat = []
    for tweet in test_x:
        y_pred = predict(tweet, freq, theta)
        y_hat.append(y_pred)

    m = len(y_hat)
    y_hat = np.array(y_hat)
    y_hat = y_hat.reshape(m)
    test_y = test_y.reshape(m)

    c = y_hat == test_y
    j = 0
    for i in c:
        if i == True:
            j += 1
    accuracy = j / m
    return y_hat, accuracy