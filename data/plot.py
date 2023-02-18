"""
Generates scatter plots of each sparse matrix as a .png
"""

import scipy.sparse as sparse
import scipy.io
import matplotlib.pyplot as plt

def fastspy(A, ax, cmap=None):
    m, n = A.shape
    #ax.hold(True)

    ax.scatter(A.col, A.row, c=A.data, s=1, marker='s',
               edgecolors='none', clip_on=False,
               cmap=cmap)

    ax.autoscale(tight=True)
    for spine in ax.spines.values():
        spine.set_position(('outward', 1))
    ax.axis('tight')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    #ax.hold(False)

if __name__ == '__main__':
    import scipy.io as io
    import glob
    import matplotlib.pyplot as plt
    for f in glob.glob("*.mtx"):
        A = scipy.io.mmread(open(f)).tocoo()
        fig = plt.figure()
        plt.tight_layout()
        ax = fig.add_subplot(111)
        fastspy(A, ax)
        plt.savefig(f.replace(".mtx", ".png"), dpi=300, bbox_inches='tight', transparent="True", pad_inches=0)
