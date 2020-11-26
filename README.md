# DSLR
DataScience and Logistic Regression python scripts

## Packages

- numpy
- matplotlib
- pandas
- seaborn

## Usage

`dataset` implies that both `test_dataset` and `train_dataset` are usable here.

Describe :
```
python3 describe.py dataset
```

Graphics :
```
python3 histogram.py train_dataset
python3 scatter_plot.py train_dataset
python3 pair_plot.py dataset
```

Logistic Regression training :
```
python3 logreg_train.py [-h] [-i ITER] [-l LEARNING] [-b BATCH] [-s] [-p] [-v]
                       train_dataset

positional arguments:
  train_dataset               select a valid train dataset

optional arguments:
  -h, --help            show this help message and exit
  -i ITER, --iter ITER  set the number of iterations (default to 1000)
  -l LEARNING, --learning LEARNING
                        set the learning rate (default to 0.1)
  -b BATCH, --batch BATCH
                        set the batch size for mini-batch gradient descent
                        algorithm
  -s, --stochastic      use the stochastic gradient descent algorithm
  -p, --precision       show the precision
  -v, --visualizer      show the resulting graphs
```

Logistic Regression prediction :
```
python3 logreg_predict.py [-h] [-s] test_dataset values

positional arguments:
  test_dataset     select a valid test dataset
  values      select a file with trained values

optional arguments:
  -h, --help  show this help message and exit
  -s, --show  show a 3D representation of the data
```
