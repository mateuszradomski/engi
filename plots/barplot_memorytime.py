import glob
import json
import os
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scienceplots

plt.style.use(['science', 'grid'])
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc}'

dirs = glob.glob("results/*")

# We track the execution time for SpMV and the required memory for storing that
# matrix.  We define k = 1/(time*memory). The heigher the constant the better
# the usage of memory for each second.
# 
# | No. | Memory | Time |   k |
# |-----|--------|------|-----|
# |   1 |    1MB |   1s |   1 |
# |   2 |   10MB | 0.5s | 0.2 |
# |   3 |    4MB | 0.1s | 2.5 |
# 
# An algorithm that takes long but uses little memory will have a high constant
# as well as a matrix that takes almost no time at all but takes a lot of
# memory
#
# Since this does not really scale between matricies, we will do the following:
# 1. Compute k for CSR
# 2. Scale everyother k in the relation to the k for CSR
# 3. This gives us relative performance between the formats for the same matrix that we can graph together

def myplot(kd, n, xticklabels, ylabel, ylim, title, outfilename):
    index = np.arange(n)
    bar_width = 0.4 /6

    fig, ax = plt.subplots(figsize=(9, 4))
    NUM_COLORS = len(kd)
    cm = plt.get_cmap('gnuplot2')
    ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    for i,k in enumerate(kd):
        ax.bar(index + i * bar_width, kd[k], bar_width, label=k)

    ax.legend(ncol=len(kd)//2)
    ax.set_xticklabels(xticklabels)
    ax.set_xticks(index + 5.5*bar_width)
    ax.set_ylim(0,ylim)
    ax.set_ylabel(ylabel)
    ax.annotate("2.46", (0.0,1.3), xycoords='data', xytext=(0.4, 1), 
     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            horizontalalignment='left', verticalalignment='top')
    fig.suptitle(title)

    plt.savefig(outfilename, dpi=300, bbox_inches='tight', pad_inches=0)

kd = {}
for directory in dirs:
    os.chdir(directory)

    size_array = []
    markers = []
    labels = []
    for i,f in enumerate(glob.glob("*json")):
        info = json.loads(open(f).read())
        time_mss = [v["time_ms"] for v in info["data"]]
        size = info["matrix_size"]

        time_ms_avg = sum(time_mss) / len(time_mss)
        memorytime = 1/(time_ms_avg * (size / (1000**2)))
        size_array.append(size / (1000 * 1000))
        label = re.findall(r'(BSR[2-8]|COO|CSC|CSR|ELL|SELL[0-9]+)', f)[0]
        if label not in kd:
            kd[label] = []
        kd[label].append(memorytime)
        labels.append(label)

    os.chdir("..\\..")

CSR_k = kd["CSR"]
scaled_ks = {}

for k in kd:
    scaled_ks[k] = [k/kCSR for k,kCSR in zip(kd[k], CSR_k)]

xticklabels = [k.replace("results\\", "")[2:] for k in dirs]
ylabel = "$\\frac{k}{k_{CSR}}$"
title = "Efektywność $k$ różnych formatów macierzy w SpMV dla wybranych macierzy"
outfilename = "plots\\barchart_memory.png"
myplot(scaled_ks, len(dirs), xticklabels, ylabel, 1.4, title, outfilename)
