import pandas as pd
import csv
import numpy as np
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

def load_dataset():
    train_data = pd.read_csv('..\\raw_data\\xtrain.csv')
    x_train = train_data.iloc[:, 1]
    y_train = train_data.iloc[:, 0]

    test_data = pd.read_csv('..\\raw_data\\balancedtest.csv')
    x_test = test_data.iloc[:, 1]
    y_test = test_data.iloc[:, 0]
    return x_train, y_train, x_test, y_test

def preprocess(text_series):
    text_series = text_series.str.lower()
    text_series = text_series.apply(lambda text: re.sub(r'\d+', '', text))  # Remove numbers
    text_series = text_series.apply(lambda text: re.sub(r'[^\w\s]', '', text))  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    text_series = text_series.apply(lambda text: ' '.join(word for word in word_tokenize(text) if word not in stop_words))
    return text_series

def train_and_save_word2vec_model(preprocessed_texts, model_filename):
    # Tokenize the sentences into words
    tokenized_data = [text.split() for text in preprocessed_texts]
    
    # Train the Word2Vec model
    model = Word2Vec(sentences=tokenized_data, vector_size=300, window=5, min_count=1, workers=4)

    # Save the model
    model.save(model_filename)
    print(f'Model saved to {model_filename}')

def main():
    print("Loading data...")
    x_train, y_train, x_test, y_test = load_dataset()
    print("Done loading data.")

    print("Preparing the data...")
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    print("Data preparation complete.")

    # Train and save the Word2Vec model
    print("Training the Word2Vec model...")
    train_and_save_word2vec_model(x_train, "word2vec_model_300d.model")
    print("Word2Vec model training and saving complete.")

if __name__ == '__main__':
    main()
