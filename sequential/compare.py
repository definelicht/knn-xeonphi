import numpy as np
import sys

def compare(labels0, labels1):
  mismatches = np.sum(labels0 != labels1)
  print("Comparing {} labels:".format(labels0.size))
  if mismatches == 0:
    print("  No mismatches detected.")
  else:
    print("  {} / {} labels mismatched ({}%)".format(
        mismatches, labels.size, mismatches/labels.size*100))

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: <first label file> <second label file>")
    sys.exit(1)
  compare(np.fromfile(sys.argv[1]), np.fromfile(sys.argv[2]))
