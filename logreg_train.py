import argparse
import sys
import math
import numpy as np
import pandas as pd
from models.Model import Model

def get_mean(a):
	a = a[~pd.isnull(a)]
	return (sum(a) / len(a))

def get_var(a):
	a = a[~pd.isnull(a)]
	if (len(a) < 2):
		return (0)
	m = get_mean(a) 
	return (sum([(ai - m)**2 for ai in a]) / (len(a) - 1))

def get_std(a):
	return (math.sqrt(get_var(a)))

def normalize(d):
	e = (d - get_mean(d)) / get_std(d)
	return (e)

def	load_file(n):
	try:
		d = pd.read_csv(n, sep = ',')
	except Exception as e:
		print ("error : {0}".format(e))
		sys.exit()
	return (d)

def logreg_train(args):
	d = load_file(args.dataset)
	try:
		# Sanitize dataset
		d = d.dropna(subset=['Herbology', 'Ancient Runes', 'Astronomy'])
		X = np.array(d.values[:, [8, 12, 7]], dtype=float)
		y = d.values[:, 1]

		# Init model
		if args.stochastic:
			args.batch = 1
		model = Model(args.iter, args.learning, args.batch > 0, args.batch, args.precision, args.visualizer)
		
		# Normalize features
		X = np.array([normalize(t) for t in X.T]).T
		new_df = pd.DataFrame(X)

		# Convert guild names to integers indexes)
		Y = []
		for i in y:
			Y.append(model.feature_i[i])
		y = np.array(Y, dtype=int)
		y_unique = np.unique(y)

		# Execute logistic regression
		model.process_logreg(X, y)
	except Exception as e:
		print ("error : {0}".format(e))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", help="select a valid dataset")
	parser.add_argument("-i", "--iter", help="set the number of iterations (default to 1000)", default=1000)
	parser.add_argument("-l", "--learning", help="set the learning rate (default to 0.1)", default=0.1)
	parser.add_argument("-b", "--batch", help="set the number of batchs for batch gradient descent algorithm", default=0)
	parser.add_argument("-s", "--stochastic", help="use the stochastic gradient descent algorithm", action="store_true")
	parser.add_argument("-p", "--precision", help="show the precision", action="store_true")
	parser.add_argument("-v", "--visualizer", help="show the resulting graphs", action="store_true")
	args = parser.parse_args()
	logreg_train(args)

if (__name__ == "__main__"):
	main()
