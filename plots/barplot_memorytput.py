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

def myplot(kd, n, xticklabels, ylabel, ylim, title, outfilename):
    index = np.arange(n)
    bar_width = 0.4 /5.5

    fig, ax = plt.subplots(figsize=(9, 4))
    NUM_COLORS = len(kd)
    cm = plt.get_cmap('gnuplot2')
    ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    for i,k in enumerate(kd):
        ax.bar(index + i * bar_width, kd[k], bar_width, label=k)

    ax.legend(ncol=(len(kd) + 1)//2)
    ax.set_xticklabels(xticklabels)
    ax.set_xticks(index + 5*bar_width)
    # ax.set_ylim(0,ylim)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)

    plt.savefig(outfilename, dpi=300, bbox_inches='tight', pad_inches=0)


kd = {}
for directory in dirs:
    os.chdir(directory)

    gflop_array = []
    size_array = []
    markers = []
    labels = []
    for i,f in enumerate(glob.glob("*json")):
        info = json.loads(open(f).read())
        time_mss = [v["time_ms"] for v in info["data"]]
        size = info["matrix_size"]

        time_ms_avg = sum(time_mss) / len(time_mss)
        memorytput = (size / (1000 * 1000 * 1000)) / (time_ms_avg / 1000)
        label = re.findall(r'(BSR[2-8]|COO|CSC|CSR|ELL|SELL[0-9]+)', f)[0]
        if label == "ELL":
            continue
        if label not in kd:
            kd[label] = []
        kd[label].append(memorytput)
        labels.append(label)

    os.chdir("..\\..")

xticklabels = [k.replace("results\\", "")[2:] for k in dirs]
ylabel = "GB/s"
title = "Całkowita przepustowość pamięci różnych formatów macierzy w SpMV dla wybranych macierzy"
outfilename = "plots\\barchart_memory_absolute_tput.png"
myplot(kd, len(dirs), xticklabels, ylabel, 300, title, outfilename)
