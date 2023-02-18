"""
Generates GFLOP x Memory usage for each matrix format in a given sparse matrix
"""

import glob
import json
import os
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import numpy as np

dirs = glob.glob("results/*")
plt.style.use(['science', 'scatter'])

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc}'

fig, ax = plt.subplots(figsize=(8, 4))
n = len(dirs)
index = np.arange(n)
bar_width = 0.4 /5

kd = {}
for directory in dirs:
    os.chdir(directory)

    gflop_array = []
    size_array = []
    availMarkers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]
    markers = []
    labels = []
    for i,f in enumerate(glob.glob("*json")):
        info = json.loads(open(f).read())
        gflops = [v["gflop"] for v in info["data"]]
        size = info["matrix_size"]

        gflop_avg = sum(gflops) / len(gflops)
        gflop_array.append(gflop_avg)
        size_array.append(size / (1000 * 1000))
        markers.append(availMarkers[i])
        label = re.findall(r'(BSR[2-8]|COO|CSC|CSR|ELL|SELL[2-8])', f)[0]

        if label not in kd:
            kd[label] = []
        kd[label].append(gflop_avg)
        labels.append(label)

    fig, ax = plt.subplots()
    for s, g, m, l in zip(size_array, gflop_array, markers, labels):
        ax.scatter(s, g, marker=m, label=l)
    #calculate equation for trendline
    z = np.polyfit(size_array, gflop_array, 1)
    p = np.poly1d(z)

    ax.set_ylabel("GFLOPS")
    ax.set_xlabel("Zużycie pamięci (MB)")
    ax.legend(ncol=2, fontsize=7)

    #add trendline to plot
    ax.plot(size_array, p(size_array), linestyle='dashed')
    print(f"Saved {f}")
    plt.savefig("..\\..\\plots\\" + directory.replace("results\\", "") + ".png", dpi=800, bbox_inches='tight', pad_inches=0)

    os.chdir("..\\..")
