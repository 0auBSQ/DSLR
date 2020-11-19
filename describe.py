import numpy
import sys
import math
import pandas as pd

def	load_file(n):
	try:
		d = pd.read_csv(n, sep = ',')
		d = d.drop(columns=["Index", "Hogwarts House", "First Name", "Birthday", "Last Name", "Best Hand"])
	except Exception as e:
		print ("error : {0}".format(e))
		sys.exit()
	return (d)

def get_count(a):
	a = a[~pd.isnull(a)]
	return (len(a))

def get_mean(a):
	a = a[~pd.isnull(a)]
	return (sum(a) / len(a))

def get_min(a):
	a = a[~pd.isnull(a)]
	a.sort()
	return (a[0])

def get_max(a):
	a = a[~pd.isnull(a)]
	a.sort()
	return (a[len(a) - 1])

def get_percentile(a, p):
	a = a[~pd.isnull(a)]
	a.sort()
	k = (len(a) - 1) * p
	f = math.floor(k)
	c = math.ceil(k)
	if (f == c):
		return (a[int(k)])
	d0 = a[int(f)] * (c - k)
	d1 = a[int(c)] * (k - f)
	return (d0 + d1)

def get_q25(a):
	return (get_percentile(a, .25))

def get_q50(a):
	return (get_percentile(a, .50))

def get_q75(a):
	return (get_percentile(a, .75))

def get_std(a):
	a = a[~pd.isnull(a)]
	if (len(a) < 2):
		return (0)
	m = get_mean(a) 
	return (math.sqrt(sum([(ai - m)**2 for ai in a]) / (len(a) - 1)))

def factorize_func(f, d):
	ret = []
	for (name, column) in d.iteritems():
		ret.append(f(column.values))
	return (ret)

def main():
	if (len(sys.argv) is not 2):
		print ("usage : python describe.py [dataset]")
		sys.exit()
	d = load_file(sys.argv[1])
	try:
		desc = pd.DataFrame(columns=d.columns.values)
		desc.loc["count"] = factorize_func(get_count, d)
		desc.loc["mean"] = factorize_func(get_mean, d)
		desc.loc["std"] = factorize_func(get_std, d)
		desc.loc["min"] = factorize_func(get_min, d)
		desc.loc["25%"] = factorize_func(get_q25, d)
		desc.loc["50%"] = factorize_func(get_q50, d)
		desc.loc["75%"] = factorize_func(get_q75, d)
		desc.loc["max"] = factorize_func(get_max, d)
		print (desc.T)
	except Exception as e:
		print ("error : {0}".format(e))

if (__name__ == "__main__"):
	main()