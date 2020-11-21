import argparse
import numpy as np
import pandas as pd

def logreg_train(args):
  df = pd.read_csv(args.dataset)

  df = df.dropna(subset=['Herbology', 'Ancient Runes', 'Astronomy'])
  X = np.array(df.values[:, [8, 12, 7]], dtype=float)
  y = df.values[:, 1]

  if args.batch > 0:
    pass

  if args.stochastic:
    pass

  if args.precision:
    pass

  if args.visualizer:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="select a valid dataset")
    parser.add_argument(
        "-i",
        "--iter",
        help="set the number of iterations (default to 1000)",
        default=1000,
    )
    parser.add_argument(
        "-b",
        "--batch",
        help="set the batch gradient descent algorithm",
        default=0,
    )
    parser.add_argument(
        "-s",
        "--stochastic",
        help="use the stochastic gradient descent algorithm",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--precision",
        help="show the precision",
        action="store_true",
    )
    parser.add_argument(
      "-v",
      "--visualizer",
      help="show the resulting graphs",
      action="store_true",
    )

    args = parser.parse_args()
    logreg_train(args)
