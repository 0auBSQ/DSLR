import numpy
import sys
import pandas as pd
import matplotlib.pyplot as plt

def factorize(d, param):
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
				s = cn
				d1tmp = factorize(d1, s)
				d2tmp = factorize(d2, s)
				d3tmp = factorize(d3, s)
				d4tmp = factorize(d4, s)
				hists[i].hist(d1tmp, 20, alpha=1, color='blue', edgecolor='black', label='Ravenclaw')
				hists[i].hist(d2tmp, 20, alpha=1, color='green', edgecolor='black', label='Slytherin')
				hists[i].hist(d3tmp, 20, alpha=1, color='red', edgecolor='black', label='Gryffindor')
				hists[i].hist(d4tmp, 20, alpha=1, color='yellow', edgecolor='black', label='Hufflepuff')
				hists[i].set_title(s)
				i += 1
		# Display histograms
		fig.tight_layout()
		plt.show()
	except Exception as e:
		print ("error : {0}".format(e))

if (__name__ == "__main__"):
	main()