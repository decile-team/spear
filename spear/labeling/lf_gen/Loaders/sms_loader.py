#!/usr/bin/env python

import numpy as np
import scipy
import json
from sklearn import model_selection as cross_validation
import re
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
def parse_file(filename):
    def parse(filename):
        tweet = []
        with open(filename) as f:
            for line in f:
                tweet.append(line)
        return tweet[1:]

    tweets = parse(filename)
    gt = []
    plots = []
    idx = []
    for i,twt in enumerate(tweets):
        tweet = twt.split(',')
        # print(tweet)
        genre = tweet[0]
        tweet_txt = re.sub(r"@\w+","", tweet[1])
        
        if 'ham' in genre:
            plots.append(tweet_txt)
            gt.append(1)
            idx.append(i)
        elif 'spam' in genre:
            plots.append(tweet_txt)
            gt.append(-1)
            idx.append(i)
        else:
            continue  

    
    return np.array(plots), np.array(gt)

def split_data(X, plots, y, split_val = 0.1):
    np.random.seed(1234)
    num_sample = np.shape(X)[0]
    num_test = 500
    X, plots, y  = shuffle(X, plots, y, random_state = 25)
    X_test = X[0:num_test,:]
    X_train = X[num_test:, :]
    plots_train = plots[num_test:]
    plots_test = plots[0:num_test]

    y_test = y[0:num_test]
    y_train = y[num_test:]

    # split dev/test
    test_ratio = split_val
    X_tr, X_te, y_tr, y_te, plots_tr, plots_te =cross_validation.train_test_split(X_train, y_train, plots_train, test_size = test_ratio, random_state=25)

    return np.array(X_tr.todense()), np.array(X_te.todense()), np.array(X_test.todense()), np.array(y_tr), np.array(y_te), np.array(y_test), plots_tr, plots_te, plots_test


class DataLoader(object):
    """ A class to load in appropriate numpy arrays
    """

    def prune_features(self, val_primitive_matrix, train_primitive_matrix, thresh=0.01):
        val_sum = np.sum(np.abs(val_primitive_matrix),axis=0)
        train_sum = np.sum(np.abs(train_primitive_matrix),axis=0)

        #Only select the indices that fire more than 1% for both datasets
        train_idx = np.where((train_sum >= thresh*np.shape(train_primitive_matrix)[0]))[0]
        val_idx = np.where((val_sum >= thresh*np.shape(val_primitive_matrix)[0]))[0]
        common_idx = list(set(train_idx) & set(val_idx))

        return common_idx

    def load_data(self, dataset, data_path='/data/sms/', split_val=0.1, feat= 'count'):
     
        plots, labels = parse_file(data_path+'all.csv')
        same = len(np.where(labels==1)[0])
        up = len(np.where(labels==-1)[0])
        upx = plots[np.where(labels==-1)[0]]
        upy = labels[np.where(labels==-1)[0]]
        times = same//up
        ly = np.tile(upy, times)
        labels  = np.concatenate((labels, ly))
        lx = np.tile(upx, times)
        plots  = np.concatenate((plots, lx))
        # print(len(plots))    
        #Featurize Plots  
        if feat == 'count':
            vectorizer = CountVectorizer(min_df=1, binary=True, stop_words='english', decode_error='ignore', strip_accents='ascii', ngram_range=(1,2))
        elif feat =='lemma':
            vectorizer = CountVectorizer(min_df=1, binary=True,   decode_error='ignore', ngram_range=(1,2) ,\
        tokenizer=LemmaTokenizer(),strip_accents = 'unicode', stop_words = 'english', lowercase = True)
        else:
            vectorizer = CountVectorizer(min_df=1, binary=True, stop_words='english', decode_error='ignore', strip_accents='ascii', ngram_range=(1,2))
        
        X = vectorizer.fit_transform(plots)
        valid_feats = np.where(np.sum(X,0)> 2)[1]
        X = X[:,valid_feats]

#         Split Dataset into Train, Val, Test
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
        train_ground, val_ground, test_ground, \
        train_plots, val_plots, test_plots = split_data(X, plots, labels, split_val)

        #Prune Feature Space
        common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)
        # print('common_idx',len(common_idx))
        # return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix[:,common_idx],             np.array(train_ground), np.array(val_ground), np.array(test_ground),train_plots, val_plots, test_plots
        return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], \
                        test_primitive_matrix[:,common_idx], np.array(train_ground), np.array(val_ground), \
                        np.array(test_ground), vectorizer, valid_feats, common_idx, \
                        train_plots, val_plots, test_plots

