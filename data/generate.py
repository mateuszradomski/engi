"""
This is used to generate dense2.mtx
"""

import random
x = 2000
y = 2000

f = open("dense2.mtx", "w")
f.write("%%MatrixMarket matrix coordinate real general\n")
f.write(f"{x} {y} {x*y}\n")

for c in range(x):
    print(c)
    for r in range(x):
        f.write(f"{c+1} {r+1} {random.random()}\n")
