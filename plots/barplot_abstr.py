import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scienceplots

plt.style.use(['science', 'grid'])

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc}'

def plot(kd, n, xticklabels, ylabel, ylim, title, outfilename):
    index = np.arange(n)
    bar_width = 0.4 /5

    fig, ax = plt.subplots(figsize=(8, 4))
    NUM_COLORS = len(kd)
    cm = plt.get_cmap('gnuplot2')
    ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    for i,k in enumerate(kd):
        ax.bar(index + i * bar_width, kd[k], bar_width, label=k)

    ax.legend(ncol=len(kd)//2)
    ax.set_xticklabels(xticklabels)
    ax.set_xticks(index + 4.5*bar_width)
    ax.set_ylim(0,ylim)
    ax.set_ylabel(ylabel)
    ax.annotate("2.46", (0.25,1.3), xycoords='data', xytext=(-0.2, 1), 
     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            horizontalalignment='left', verticalalignment='top')
    fig.suptitle(title)

    plt.savefig(outfilename, dpi=300, bbox_inches='tight', pad_inches=0)
