import matplotlib.pyplot as plt

nCores = []
flann = {"build": [], "search": []}
randomized = {"build": [], "search": []}

with open("euler.txt", "r") as inFile:
  for line in inFile:
    [cores, method, elapsed] = line.split(",")
    nCores.append(int(cores))
    if method == "flannBuild":
      flann["build"].append(elapsed)
    elif method == "flannSearch":
      flann["search"].append(elapsed)
    elif method == "randomizedBuild":
      randomized["build"].append(elapsed)
    elif method == "randomizedSearch":
      randomized["search"].append(elapsed)

print(flann)
print(randomized)
