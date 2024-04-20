import pandas as pd
import csv
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from lstmclassifier import train_and_evaluate_model, LSTMClassifier
from otherclassifier import train_and_evaluate_rf, train_and_evaluate_svm, train_and_evaluate_xgboost, train_and_evaluate_catboost, train_and_evaluate_stacking, train_and_evaluate_majority_vote



device = torch.device("cpu")
print(f"Using device: {device}")

def Load_dataset():
    train_data = pd.read_csv('..\\raw_data\\xtrain.csv')
    # train_data = pd.read_csv('..\\raw_data\\fulltrain.csv')
    x_train = train_data.iloc[:, 1]
    y_train = train_data.iloc[:, 0] - 1

    test_data = pd.read_csv('..\\raw_data\\balancedtest.csv')
    x_test = test_data.iloc[:, 1]
    y_test = test_data.iloc[:, 0] - 1
    return x_train, y_train, x_test, y_test


def Preprocess(text_series):
    # Convert text to lowercase
    text_series = text_series.str.lower()
    text_series = text_series.apply(lambda text: re.sub(r'\d+', '', text))  # Remove numbers
    text_series = text_series.apply(lambda text: re.sub(r'[^\w\s]', '', text))  # Remove punctuation
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text_series = text_series.apply(lambda text: ' '.join(word for word in word_tokenize(text) if word not in stop_words))
    
    # Initialize stemmer & Apply stemming
    # stemmer = PorterStemmer()
    # text_series = text_series.apply(lambda text: ' '.join([stemmer.stem(word) for word in word_tokenize(text)]))
    
    # Initialize lemmatizer & Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    text_series = text_series.apply(lambda text: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)]))
    
    return text_series


def Pretrained_W2V_FE(Series, w2v_model):
    vec = []
    for s in Series:
        s_vec = [w2v_model[token] for token in s if token in w2v_model]
        if len(s_vec) == 0:
            vec.append(np.zeros(100))
        else:
            vec.append(np.mean(s_vec, axis = 0))
    return vec

def Selftrained_W2V_FE(series, w2v_model):
    vec = []
    for s in series:
        words = s.split()
        s_vec = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if len(s_vec) == 0:
            vec.append(np.zeros(w2v_model.vector_size))
        else:
            vec.append(np.mean(s_vec, axis=0))
    return vec

def sequence_W2V_FE(series, w2v_model, max_len=100):
    vec_sequence = []
    for s in series:
        words = s.split()
        sequence = [w2v_model.wv[word] for word in words if word in w2v_model.wv][:max_len]
        sequence += [np.zeros(w2v_model.vector_size)] * (max_len - len(sequence))  # Padding
        vec_sequence.append(np.array(sequence)) 
    return np.stack(vec_sequence)  


def load_and_extract_features(model_filename, preprocessed_texts):
    model = Doc2Vec.load(model_filename)
    features = [model.infer_vector(text.split()) for text in preprocessed_texts]
    return features

def tfidf_feature_engineering(text_series, max_features=5000):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_features = tfidf_vectorizer.fit_transform(text_series)
    return tfidf_features, tfidf_vectorizer

def bow_feature_engineering(text_series, max_features=5000):
    bow_vectorizer = CountVectorizer(max_features=max_features)
    bow_features = bow_vectorizer.fit_transform(text_series)
    return bow_features, bow_vectorizer

def main():
    
    print("loading data...")
    x_train, y_train, x_test, y_test = Load_dataset()
    print("done")

    print("Preparing the data...")
    x_train = Preprocess(x_train)
    x_test = Preprocess(x_test)
    print("done")

    print("Vectorising the text...")
    # word2vec(Pre-trained)
    # w2v_model = KeyedVectors.load_word2vec_format('..\\word2vec_model\\pre_trained\\word2vec.glove.6B.100d.txt')
    # w2v_model = KeyedVectors.load_word2vec_format('..\\word2vec_model\\pre_trained\\glove.6B.200d.Word2vec.txt')
    # w2v_model = KeyedVectors.load_word2vec_format('..\\word2vec_model\\pre_trained\\glove.42B.300d.word2vec.txt')
    # vec_train = np.array(Pretrained_W2V_FE(x_train, w2v_model))
    # vec_test = np.array(Pretrained_W2V_FE(x_test, w2v_model))

    # word2vec(Self-trained)
    # w2v_model = Word2Vec.load('..\\word2vec_model\\self_trained\\word2vec_model.model')
    # w2v_model = Word2Vec.load('..\\word2vec_model\\self_trained\\word2vec_model_300d.model')
    # vec_train = np.array(sequence_W2V_FE(x_train, w2v_model))
    # vec_test = np.array(sequence_W2V_FE(x_test, w2v_model))
    # vec_train = np.array(Selftrained_W2V_FE(x_train, w2v_model))
    # vec_test = np.array(Selftrained_W2V_FE(x_test, w2v_model))


    # print(vec_train.shape)
    # BoW
    vec_train, Bow_vectorizer = bow_feature_engineering(x_train)
    vec_test = Bow_vectorizer.transform(x_test)

    # doc2vec
    # vec_train = load_and_extract_features("..\\doc2vec_model\\doc2vec_model.model", x_train)
    # vec_test = load_and_extract_features("..\\doc2vec_model\\doc2vec_model.model", x_test)

    # TF-IDF
    # vec_train, tfidf_vectorizer = tfidf_feature_engineering(x_train)
    # vec_test = tfidf_vectorizer.transform(x_test)

    print("done")

    # define 
    print("Building model & Training...")
    # LSTM
    # train_and_evaluate_model(vec_train, y_train, vec_test, y_test, device)

    # SVM
    # train_and_evaluate_svm(vec_train, y_train, vec_test, y_test)

    # Random Forest
    train_and_evaluate_rf(vec_train, y_train, vec_test, y_test)

    # xgboost
    # train_and_evaluate_xgboost(vec_train, y_train, vec_test, y_test)
    
    # catboost
    # train_and_evaluate_catboost(vec_train, y_train, vec_test, y_test)
    
    # stacking for 4 model
    # train_and_evaluate_stacking(vec_train, y_train, vec_test, y_test)

    # majority_vote for 4 model
    # train_and_evaluate_majority_vote(vec_train, y_train, vec_test, y_test)

    print("done")

if __name__ =='__main__':
    
    main()