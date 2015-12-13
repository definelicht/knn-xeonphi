import numpy as np
import re, sys

if len(sys.argv) < 3:
  print("Please provide input and output file paths.")
  sys.exit(1)

nCores = [1, 2, 4, 8, 16, 24]
stride = 6
our = [[] for i in range(stride)]
flann = [[] for i in range(stride)]
with open(sys.argv[1]) as inFile:
  inFile.readline()
  done = False
  while not done:
    for i in range(stride):
      line = inFile.readline()
      if line == "":
        done = True
        break
      match = re.search("([0-9]+)[ \t]+([0-9]+\.[0-9]+)[ \t]+([0-9]+\.[0-9]+)",
                        line)
      if match:
        match = match.groups()
        our[i].append(float(match[1]))
        flann[i].append(float(match[2]))

resultOur = {}
resultFlann = {}
for (result, values) in [(resultOur, our), (resultFlann, flann)]:
  result["mean"] = np.empty(stride, dtype=np.float)
  result["std"] = np.empty(stride, dtype=np.float)
  result["error"] = np.empty(stride, dtype=np.float)
  result["speedup"] = np.empty(stride, dtype=np.float)
  result["speedupStd"] = np.empty(stride, dtype=np.float)
  result["speedupError"] = np.empty(stride, dtype=np.float)
  baseline = np.array(values[0])
  for i in range(stride):
    arr = np.array(values[i])
    result["mean"][i] = np.mean(arr)
    result["std"][i] = np.std(arr)
    result["error"][i] = result["std"][i]/np.sqrt(arr.size)
    result["speedup"][i] = result["mean"][0]/result["mean"][i]
    result["speedupStd"][i] = abs(result["speedup"][i]) * ((result["error"][i]/result["mean"][i])**2 + (result["error"][0]/result["mean"][0])**2)**0.5
    result["speedupError"][i] = result["speedupStd"][i]/np.sqrt(arr.size)

speedup = resultFlann["mean"]/resultOur["mean"]
speedupError = (np.abs(speedup) *
                np.sqrt((resultOur["error"]/resultOur["mean"])**2 +
                        (resultFlann["error"]/resultFlann["mean"])**2))

with open(sys.argv[2], "w") as outFile:
  for i in range(stride):
    outFile.write("{},{},{},{},{},{},{},{},{}\n".format(
      nCores[i], resultOur["mean"][i], resultOur["std"][i],
      resultOur["error"][i], resultFlann["mean"][i], resultFlann["std"][i],
      resultFlann["error"][i], speedup[i], speedupError[i]))

