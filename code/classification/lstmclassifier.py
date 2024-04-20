import pandas as pd
import csv
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import KFold


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        num_layers = 2
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.hidden_dim, device=device),
                torch.zeros(1, batch_size, self.hidden_dim, device=device))

    def forward(self, features):
        # feature: (batch_size, sequence_length, embedding_dim)
        lstm_out, _ = self.lstm(features)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores

    
def train_and_evaluate_model(vec_train , y_train, vec_test, y_test, device):
    embedding_dim = 200
    model = LSTMClassifier(embedding_dim=embedding_dim,hidden_dim = 32,tagset_size = 4).to(device) #classification
    # initialize tensor
    X_tensor = torch.tensor(vec_train, dtype = torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype = torch.long)
    # if X_tensor.ndim == 2:
    #     X_tensor = X_tensor.unsqueeze(1)
    n_splits = 5
    num_epochs = 30
    kf = KFold(n_splits=n_splits, shuffle=True)

    for fold, (train_index, val_index) in enumerate(kf.split(X_tensor), 1):
        print(f'Fold {fold}/{n_splits}')

        X_train_fold = X_tensor[train_index]
        y_train_fold = y_tensor[train_index]
        X_val_fold = X_tensor[val_index]
        y_val_fold = y_tensor[val_index]
        
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        model = LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=32, tagset_size=4).to(device)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.05)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                model.hidden = model.init_hidden(X_batch.size(0), device)
                optimizer.zero_grad()
                tag_scores = model(X_batch)
                loss = loss_function(tag_scores, y_batch)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for X_batch, y_batch in val_loader:
                    tag_scores = model(X_batch)
                    val_loss += loss_function(tag_scores, y_batch).item()
                val_loss /= len(val_loader)
                print(f'Validation loss: {val_loss}')

    # Test() 
    print("Starting test...")
    # X_test_tensor = torch.tensor(vec_test, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(vec_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
    f1 = f1_score(y_test_tensor.numpy(),predicted.numpy(),average="weighted")
    print(f"F1 Score (Weighted): {f1}")
    
    print(classification_report(y_test_tensor.cpu().numpy(), predicted.cpu().numpy()))
    print("done")

