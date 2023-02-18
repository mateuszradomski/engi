import glob
import json
import os
import re

dirs = glob.glob("results/*")
max_error = 0.0
max_error_format = None
max_error_filename = None

for directory in dirs:
    os.chdir(directory)
    for i,f in enumerate(glob.glob("*json")):
        info = json.loads(open(f).read())
        label = re.findall(r'(BSR[2-8]|COO|CSC|CSR|ELL|SELL[2-8])', f)[0]
        if max_error < float(info["max_error"]):
            max_error = float(info["max_error"])
            max_error_format = label
            max_error_filename = f
    os.chdir("..\\..")

print(max_error, max_error_format, max_error_filename)
