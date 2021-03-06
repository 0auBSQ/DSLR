import numpy
import sys
import pandas as pd
import matplotlib.pyplot as plt

def factorize(d, param):
	# Transform dataframe parameter column to nan-free numpy list
	dtmp = d[[param]].T.to_numpy()
	return (dtmp[~pd.isnull(dtmp)])

def	load_file(n):
	try:
		d = pd.read_csv(n, sep = ',')
		d = d.drop(columns=["Index", "First Name", "Birthday", "Last Name", "Best Hand"])
	except Exception as e:
		print ("error : {0}".format(e))
		sys.exit()
	return (d)

def main():
	if (len(sys.argv) is not 2):
		print ("usage : python histogram.py [dataset]")
		sys.exit()
	d = load_file(sys.argv[1])
	try:
		# Split dataframe by house
		d1 = d.loc[d['Hogwarts House'] == 'Ravenclaw']
		d2 = d.loc[d['Hogwarts House'] == 'Slytherin']
		d3 = d.loc[d['Hogwarts House'] == 'Gryffindor']
		d4 = d.loc[d['Hogwarts House'] == 'Hufflepuff']
		# Init hists
		fig, axes = plt.subplots(nrows=4, ncols=4)
		hists = axes.flatten()
		# Build histograms
		i = 0
		for (cn, cd) in d.iteritems():
			if (cn != 'Hogwarts House' and i < 15):
				hists[i].hist(factorize(d1, cn), 20, alpha=1, color='blue', edgecolor='black', label='Ravenclaw')
				hists[i].hist(factorize(d2, cn), 20, alpha=1, color='green', edgecolor='black', label='Slytherin')
				hists[i].hist(factorize(d3, cn), 20, alpha=1, color='red', edgecolor='black', label='Gryffindor')
				hists[i].hist(factorize(d4, cn), 20, alpha=1, color='yellow', edgecolor='black', label='Hufflepuff')
				hists[i].set_title(cn)
				i += 1
		# Display histograms
		fig.tight_layout()
		plt.show()
	except Exception as e:
		print ("error : {0}".format(e))

if (__name__ == "__main__"):
	main()