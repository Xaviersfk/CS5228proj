import pandas as pd
import csv
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from lstmclassifier import train_and_evaluate_model, LSTMClassifier

device = torch.device("cuda")
print(f"Using device: {device}")

def Load_dataset():
    train_data = pd.read_csv('..\\raw_data\\xtrain.csv')
    # train_data = pd.read_csv('..\\raw_data\\fulltrain.csv')
    x_train = train_data.iloc[:, 1]
    y_train = train_data.iloc[:, 0]

    test_data = pd.read_csv('..\\raw_data\\balancedtest.csv')
    x_test = test_data.iloc[:, 1]
    y_test = test_data.iloc[:, 0]
    return x_train, y_train, x_test, y_test


def Preprocess(text_series):
    # Convert text to lowercase
    text_series = text_series.str.lower()
    text_series = text_series.apply(lambda text: re.sub(r'\d+', '', text))  # Remove numbers
    text_series = text_series.apply(lambda text: re.sub(r'[^\w\s]', '', text))  # Remove punctuation
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text_series = text_series.apply(lambda text: ' '.join(word for word in word_tokenize(text) if word not in stop_words))
    
    return text_series


def Feature_engineering(Series, w2v_model):
    vec = []
    for s in Series:
        s_vec = [w2v_model[token] for token in s if token in w2v_model]
        if len(s_vec) == 0:
            vec.append(np.zeros(100))
        else:
            vec.append(np.mean(s_vec, axis = 0))
    return vec

def main():
    

    print("loading data...")
    x_train, y_train, x_test, y_test = Load_dataset()
    print("done")

    print("Preparing the data...")
    x_train = Preprocess(x_train)
    x_test = Preprocess(x_test)
    print("done")

    print("Vectorising the text...")
    w2v_model = KeyedVectors.load_word2vec_format('word2vec.glove.6B.100d.txt')
    vec_train = np.array(Feature_engineering(x_train, w2v_model))
    vec_test = np.array(Feature_engineering(x_test, w2v_model))
    print("done")

    # define 
    print("Building model & Training...")
    train_and_evaluate_model(vec_train,y_train, vec_test,y_test,device)
    print("done")

if __name__ =='__main__':
    
    main()