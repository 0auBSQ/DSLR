import numpy
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt

def factorize(d, param):
	# Transform dataframe parameter column to nan-free numpy list
	dtmp = d[[param]].T.to_numpy()
	return (dtmp[~pd.isnull(dtmp)])

def get_var(a):
	a = a[~pd.isnull(a)]
	if (len(a) < 2):
		return (0)
	m = get_mean(a) 
	return (sum([(ai - m)**2 for ai in a]) / (len(a) - 1))

def get_std(a):
	return (math.sqrt(get_var(a)))

def get_mean(a):
	a = a[~pd.isnull(a)]
	return (sum(a) / len(a))

def get_cov(a, b):
	return (sum(((a - get_mean(a)) * (b - get_mean(b))) / (len(a) - 1)))
	
def get_pcc(a, b):
	return (get_cov(a, b) / (get_std(a) * get_std(b)))

def	load_file(n):
	try:
		d = pd.read_csv(n, sep = ',')
		d = d.drop(columns=["Index", "Hogwarts House", "First Name", "Birthday", "Last Name", "Best Hand"])
	except Exception as e:
		print ("error : {0}".format(e))
		sys.exit()
	return (d)

def main():
	if (len(sys.argv) is not 2):
		print ("usage : python scatter_plot.py [dataset]")
		sys.exit()
	d = load_file(sys.argv[1])
	try:
		d = d.dropna()
		# Check Pearson's correlation coefficient of each combinaison
		ret = pd.DataFrame(columns=["pcc", "comb"])
		i = 0
		for (cn, cv) in d.iteritems():
			j = 0
			for (dn, dv) in d.iteritems():
				if (dn != cn and j > i):
					ret.loc["[" + cn + "|" + dn + "]", "pcc"] = abs(get_pcc(factorize(d, cn), factorize(d, dn)))		
					ret.loc["[" + cn + "|" + dn + "]", "comb"] = "[" + cn + "|" + dn + "]"
				j += 1
			i += 1
		# Display scatterplot
		ret.plot.scatter(x="comb", y="pcc", c="pcc", colormap="Spectral")
		plt.subplots_adjust(bottom=0.5)
		plt.xticks(rotation='vertical')
		plt.title("Correlation between attributes using Pearson's correlation coefficient")
		plt.show()
	except Exception as e:
		print ("error : {0}".format(e))

if (__name__ == "__main__"):
	main()