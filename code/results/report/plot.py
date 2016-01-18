import numpy
# import matplotlib.pyplot as plt

nCores = []
flann = {"build": [], "search": []}
randomized = {"build": [], "search": []}

with open("euler.txt", "r") as inFile:
  for line in inFile:
      sline = line.rstrip()
      [cores, method, elapsed] = sline.split(",")
      nCores.append(int(cores))
      if method == "flannBuild":
          flann["build"].append(float(elapsed))
      elif method == "flannSearch":
          flann["search"].append(float(elapsed))
      elif method == "randomizedBuild":
          randomized["build"].append(float(elapsed))
      elif method == "randomizedSearch":
          randomized["search"].append(float(elapsed))

out_fb = open("flannBuild.dat", "w")
out_fs = open("flannSearch.dat", "w")
out_rb = open("randomizedBuild.dat", "w")
out_rs = open("randomizedSearch.dat", "w")
out_sb = open("buildSpeedup.dat", "w")
out_ss = open("searchSpeedup.dat", "w")
for i in range(9):
    core = nCores[i*12]
    fb = numpy.array(flann["build"][i*3:i*3+3])
    fs = numpy.array(flann["search"][i*3:i*3+3])
    rb = numpy.array(randomized["build"][i*3:i*3+3])
    rs = numpy.array(randomized["search"][i*3:i*3+3])
    sb = numpy.array([0.0,0.0,0.0])
    ss = numpy.array([0.0,0.0,0.0])
    for j in range(3):
        sb[j] = fb[j]/rb[j]
        ss[j] = fs[j]/rs[j]
    out_fb.write("%e\t%e\t%e\n" % (core, numpy.mean(fb), numpy.std(fb)))
    out_fs.write("%e\t%e\t%e\n" % (core, numpy.mean(fs), numpy.std(fs)))
    out_rb.write("%e\t%e\t%e\n" % (core, numpy.mean(rb), numpy.std(rb)))
    out_rs.write("%e\t%e\t%e\n" % (core, numpy.mean(rs), numpy.std(rs)))
    out_sb.write("%e\t%e\t%e\n" % (core, numpy.mean(sb), numpy.std(sb)))
    out_ss.write("%e\t%e\t%e\n" % (core, numpy.mean(ss), numpy.std(ss)))
out_fb.close()
out_fs.close()
out_rb.close()
out_rs.close()
out_sb.close()
out_ss.close()

# print numpy.array(flann["build"])

# print(flann)
# print(randomized)
