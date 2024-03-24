import pandas as pd
import csv
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import KFold
device = torch.device("cpu")
print(f"Using device: {device}")

def Load_dataset():
	train_data = pd.read_csv('..\\raw_data\\fulltrain.csv')
	x_train = train_data.iloc[:, 1]
	y_train = train_data.iloc[:, 0]

	test_data = pd.read_csv('..\\raw_data\\balancedtest.csv')
	x_test = test_data.iloc[:, 1]
	y_test = test_data.iloc[:, 0]
	return x_train, y_train, x_test, y_test


def Preprocess(text_series):
    # Convert text to lowercase
    text_series = text_series.str.lower()
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


class LSTMClassifier(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, tagset_size):
		super(LSTMClassifier, self).__init__()
		self.hidden_dim = hidden_dim
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 线性层将隐藏状态空间映射到标注空间
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

	def init_hidden(self, batch_size, device):
		return (torch.zeros(1, batch_size, self.hidden_dim, device=device),
				torch.zeros(1, batch_size, self.hidden_dim, device=device))

	def forward(self, features):
		# 初始化隐藏状态
		lstm_out,_ = self.lstm(features)
		tag_space = self.hidden2tag(lstm_out[:, -1, :])
		tag_scores = nn.functional.log_softmax(tag_space, dim=1)
		return tag_scores
	

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
	model = LSTMClassifier(embedding_dim=100,hidden_dim = 32,tagset_size = 4).to(device) #classification
	# initialize tensor
	y_train = y_train - 1
	X_tensor = torch.tensor(vec_train, dtype = torch.float32)
	y_tensor = torch.tensor(y_train.values, dtype = torch.long)
	if X_tensor.ndim == 2:
		X_tensor = X_tensor.unsqueeze(1)
	# Train
	n_splits = 5
	num_epochs = 100
	kf = KFold(n_splits=n_splits, shuffle=True)

	for fold, (train_index, val_index) in enumerate(kf.split(X_tensor), 1):
		print(f'Fold {fold}/{n_splits}')
		# 分割数据
		X_train_fold = X_tensor[train_index]
		y_train_fold = y_tensor[train_index]
		X_val_fold = X_tensor[val_index]
		y_val_fold = y_tensor[val_index]
		# 创建PyTorch数据加载器
		train_dataset = TensorDataset(X_train_fold, y_train_fold)
		val_dataset = TensorDataset(X_val_fold, y_val_fold)
		train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=16)
		# 实例化模型和优化器
		model = LSTMClassifier(embedding_dim=100, hidden_dim=32, tagset_size=4).to(device)
		loss_function = nn.NLLLoss()
		optimizer = optim.SGD(model.parameters(), lr=0.02)
		# 训练过程
		for epoch in range(num_epochs):
			model.train()
			for X_batch, y_batch in train_loader:
				X_batch, y_batch = X_batch.to(device), y_batch.to(device)
				model.hidden = model.init_hidden(X_batch.size(0), device)
				# 清除梯度
				optimizer.zero_grad()
				# 执行前向传播
				tag_scores = model(X_batch)
				# 计算损失
				loss = loss_function(tag_scores, y_batch)
				# 反向传播和优化
				loss.backward()
				optimizer.step()
				
			# 验证过程
			model.eval()
			with torch.no_grad():
				val_loss = 0
				for X_batch, y_batch in val_loader:
					tag_scores = model(X_batch)
					val_loss += loss_function(tag_scores, y_batch).item()
				val_loss /= len(val_loader)
				print(f'Validation loss: {val_loss}')
	print("done")


	# Test() 
	print("Starting test...")
	X_test_tensor = torch.tensor(vec_test, dtype=torch.float32).unsqueeze(1).to(device)  # 添加序列长度维度
	y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device) - 1 
	model.eval()
	with torch.no_grad():
		outputs = model(X_test_tensor)
		_, predicted = torch.max(outputs, 1)
	f1 = f1_score(y_test_tensor.numpy(),predicted.numpy(),average="weighted")
	print(f"F1 Score (Weighted): {f1}")
	# 打印完整的分类报告，包括精确度、召回率和F1得分
	print(classification_report(y_test_tensor.cpu().numpy(), predicted.cpu().numpy()))
	print("done")

if __name__ =='__main__':
	
    main()