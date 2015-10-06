#!/usr/bin/env python3
import numpy as np
import sklearn.neighbors
import sys

def knn(k, training, labels, test):
  model = sklearn.neighbors.KNeighborsClassifier(k, weights="distance")
  model.fit(training, labels)
  return model.predict(test)

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Please specify k.")
    sys.exit(1)
  k = int(sys.argv[1])
  training = np.fromfile("dummyTraining.data").reshape([16, 3])
  test = np.fromfile("dummyTest.data").reshape([10000, 2])
  knn(k, training[:, :-1], training[:, -1], test).tofile(
      "dummyReference_{}.data".format(k))

