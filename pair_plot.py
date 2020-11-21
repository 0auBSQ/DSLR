import numpy
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
		print ("usage : python pair_plot.py [dataset]")
		sys.exit()
	d = load_file(sys.argv[1])
	try:
		g = sns.pairplot(d, hue="Hogwarts House", vars=numpy.delete(d.columns.to_numpy(), [0]))
		for ax in g.axes.flatten():
			ax.set_xlabel(ax.get_xlabel(), rotation=15)
			ax.set_ylabel(ax.get_ylabel(), rotation=45)
			ax.yaxis.get_label().set_horizontalalignment('right')
		plt.subplots_adjust(left=0.11, right=0.94, bottom=0.12, top=0.94)
		plt.show()
	except Exception as e:
		print ("error : {0}".format(e))

if (__name__ == "__main__"):
	main()