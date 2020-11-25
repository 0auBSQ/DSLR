import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Model:
	def __init__(self, epoch=1000, learning=0.1, batch=False, nBatch=10, verbose=False, visu=False):
		self.epoch = int(epoch)
		self.learning = float(learning)
		self.batch = bool(batch) 
		self.nBatch = int(nBatch) 
		self.verbose = bool(verbose) 
		self.visu = bool(visu)
		self.feature_i = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}
		self.i_feature = {0: 'Ravenclaw', 1: 'Slytherin', 2: 'Gryffindor', 3: 'Hufflepuff'}
		self.cost = [[],[],[],[]]
		self.ret = pd.DataFrame(columns=['One', 'Herbology', 'Ancient Runes', 'Astronomy'])

	def sigmoid(self, z):
		return ((1 / (1 + np.exp(-z))))
	
	def hypothesis(self, theta, X):
		return (self.sigmoid(np.dot(X, theta)))

	def loss(self, theta, X, y):
		h = self.hypothesis(theta, X)
		return ((-1 / X.shape[0]) * np.sum((y * np.log(h)) + ((1 - y) * np.log(1 - h))))

	def expandList(self, lst): 
		return list(map(lambda el:[el], lst)) 
	
	def gradiant_descent(self, X, y):
		X = np.insert(X, 0, 1, axis=1)
		for h in np.unique(y):
			if (self.verbose):
				print("Now processing : " + self.i_feature[h])
			y_h = np.array(self.expandList(np.where(y == h, 1, 0)))
			self.theta = np.zeros((X.shape[1], 1))
			for i in range(self.epoch):
				grad = np.dot(X.T, (self.hypothesis(self.theta, X) - y_h)) / len(y_h)
				self.theta -= (self.learning * grad)
				self.cost[h].append(self.loss(self.theta, X, y_h))
				if (self.verbose and self.epoch >= 10 and (i + 1) % (self.epoch // 10) == 0):
					print("Epoch " + str(i + 1) + " / " + str(self.epoch) + " : Loss : " + str(self.loss(self.theta, X, y_h)))
			self.ret.loc[self.i_feature[h]] = [item for sublist in self.theta for item in sublist]
	
	def create_batches(self, X, y, size):
		for e in np.arange(0, X.shape[0], size):
			yield X[e:e + size], y[e:e + size]
	
	def batch_gradiant_descent(self, X, y):
		X = np.insert(X, 0, 1, axis=1)
		for h in np.unique(y):
			if (self.verbose):
				print("Now processing : " + self.i_feature[h])
			y_h = np.array(self.expandList(np.where(y == h, 1, 0)))
			self.theta = np.zeros((X.shape[1], 1))
			for i in range(self.epoch):
				for X_chunk, y_chunk in self.create_batches(X, y_h, self.nBatch):
					grad = np.dot(X_chunk.T, (self.hypothesis(self.theta, X_chunk) - y_chunk)) / len(y_chunk)
					self.theta -= (self.learning * grad)
					self.cost[h].append(self.loss(self.theta, X_chunk, y_chunk))
					if (self.verbose and self.epoch >= 10 and (i + 1) % (self.epoch // 10) == 0):
						print("Epoch " + str(i + 1) + " / " + str(self.epoch) + " : Loss : " + str(self.loss(self.theta, X_chunk, y_chunk)))	
			self.ret.loc[self.i_feature[h]] = [item for sublist in self.theta for item in sublist]
	
	def display_cost(self):
		for h in range(len(self.feature_i)):
			plt.plot(self.cost[h], label=self.i_feature[h])
		plt.legend(list(self.feature_i.keys()))
		plt.title("Logistic Regression loss function evolution")
		plt.show()	
	
	def process_logreg(self, X, y):
		if (self.batch == False):
			self.gradiant_descent(X, y)
		else:
			self.batch_gradiant_descent(X, y)
		if (self.visu == True):
			self.display_cost()
		self.ret.to_csv("output.csv", index_label='Hogwarts House', header=['One', 'Herbology', 'Ancient Runes', 'Astronomy'])
		print("output.csv successfully writen !")