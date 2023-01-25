import glob
import json
import os
import matplotlib.pyplot as plt

dirs = glob.glob("results/*")


for directory in dirs:
    os.chdir(directory)

    gflop_array = []
    size_array = []
    for f in glob.glob("*json"):
        info = json.loads(open(f).read())
        gflops = [v["gflop"] for v in info["data"]]
        size = info["matrix_size"]

        gflop_array.append(sum(gflops) / len(gflops))
        size_array.append(size)

    plt.scatter(size_array, gflop_array)
    plt.show()
    os.chdir("..\\..")
    
