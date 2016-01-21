import numpy as np
# import matplotlib.pyplot as plt

flann = {"build": {}, "search": {}}
randomized = {"build": {}, "search": {}}

with open("euler.txt", "r") as inFile:
  for line in inFile:
    sline = line.rstrip()
    [cores, method, elapsed] = sline.split(",")
    if method == "flannBuild":
      if not cores in flann["build"]:
        flann["build"][cores] = []
      flann["build"][cores].append(float(elapsed))
    elif method == "flannSearch":
      if not cores in flann["search"]:
        flann["search"][cores] = []
      flann["search"][cores].append(float(elapsed))
    elif method == "randomizedBuild":
      if not cores in randomized["build"]:
        randomized["build"][cores] = []
      randomized["build"][cores].append(float(elapsed))
    elif method == "randomizedSearch":
      if not cores in randomized["search"]:
        randomized["search"][cores] = []
      randomized["search"][cores].append(float(elapsed))

for (filename, dictionary) in [("flannBuild.dat", flann["build"]),
                               ("randomizedBuild.dat", randomized["build"]),
                               ("flannSearch.dat", flann["search"]),
                               ("randomizedSearch.dat", randomized["search"])]:
  with open(filename, "w") as outFile:
    for cores in sorted(dictionary, key=int):
      outFile.write("%i\t%e\t%e\n" % (int(cores),
                                      np.mean(dictionary[cores]),
                                      np.std(dictionary[cores])
                                      /np.sqrt(len(dictionary[cores]))))

for method in ["build", "search"]:
  with open("{}Speedup.dat".format(method), "w") as outFile:
    for cores in sorted(flann[method], key=int):
      flannMean = np.mean(flann[method][cores])
      flannStd = np.std(flann[method][cores])
      ourMean = np.mean(randomized[method][cores])
      ourStd = np.std(randomized[method][cores])
      speedup = flannMean/ourMean
      error = speedup * ((flannStd/flannMean)**2 + (ourStd/ourMean)**2)**.5
      outFile.write("%i\t%e\t%e\n" % (int(cores), speedup, error))

# out_fb = open("flannBuild.dat", "w")
# out_fs = open("flannSearch.dat", "w")
# out_rb = open("randomizedBuild.dat", "w")
# out_rs = open("randomizedSearch.dat", "w")
# out_sb = open("buildSpeedup.dat", "w")
# out_ss = open("searchSpeedup.dat", "w")
# for i in range(9):
#     core = nCores[i*12]
#     fb = numpy.array(flann["build"][i*3:i*3+3])
#     fs = numpy.array(flann["search"][i*3:i*3+3])
#     rb = numpy.array(randomized["build"][i*3:i*3+3])
#     rs = numpy.array(randomized["search"][i*3:i*3+3])
#     sb = numpy.array([0.0,0.0,0.0])
#     ss = numpy.array([0.0,0.0,0.0])
#     for j in range(3):
#         sb[j] = fb[j]/rb[j]
#         ss[j] = fs[j]/rs[j]
#     out_fb.write("%e\t%e\t%e\n" % (core, numpy.mean(fb), numpy.std(fb)))
#     out_fs.write("%e\t%e\t%e\n" % (core, numpy.mean(fs), numpy.std(fs)))
#     out_rb.write("%e\t%e\t%e\n" % (core, numpy.mean(rb), numpy.std(rb)))
#     out_rs.write("%e\t%e\t%e\n" % (core, numpy.mean(rs), numpy.std(rs)))
#     out_sb.write("%e\t%e\t%e\n" % (core, numpy.mean(sb), numpy.std(sb)))
#     out_ss.write("%e\t%e\t%e\n" % (core, numpy.mean(ss), numpy.std(ss)))
# out_fb.close()
# out_fs.close()
# out_rb.close()
# out_rs.close()
# out_sb.close()
# out_ss.close()

# print numpy.array(flann["build"])

# print(flann)
# print(randomized)
