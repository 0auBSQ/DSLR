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

def write_prediction(houses):
    try:
        df = pd.DataFrame({"Hogwarts House": houses})
        df.to_csv("./houses.csv", mode="w", index=True, index_label="Index")
    except:
        print(f"Something goes wrong while writing prediction results", file=sys.stderr)
        sys.exit(-1)

def logreg_predict(args):
  d = load_file(args.dataset)
  v = load_file(args.values)
  
  try:
		# Sanitize dataset
    d = d.fillna(0)

		# Normalize features
    X = np.array(d.values[:, [8, 12, 7]], dtype=float)
    X = np.array([normalize(t) for t in X.T]).T
    X = np.insert(X, 0, 1, axis=1)
    
    theta = np.array(v.values[:, 1:].T, dtype=float)
    
    model = Model()
    prediction = model.hypothesis(theta, X)

		# Convert integers indexes to guild names)
    houses = np.argmax(prediction, axis=1)
    matching_houses = list(map(lambda v: model.i_feature[v], houses))

    write_prediction(matching_houses)
    print("houses.csv successfully written !")
  except Exception as e:
    print ("error : {0}".format(e))
    sys.exit(-1)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", help="select a valid dataset")
  parser.add_argument("values", help="select a file with trained values")
  args = parser.parse_args()
  logreg_predict(args)

if __name__ == "__main__":
  main()
