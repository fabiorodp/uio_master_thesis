# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import os
from helper import loadResults, plotPies, topWorstBest

ROOT_DIR = os.path.abspath(os.curdir)

files = [
    f"{ROOT_DIR}/results/WINJ21_60min.json",
    f"{ROOT_DIR}/results/WINM21_60min.json",
    f"{ROOT_DIR}/results/WINQ21_60min.json",
    f"{ROOT_DIR}/results/WINV21_60min.json",
    f"{ROOT_DIR}/results/WINZ21_60min.json",
    f"{ROOT_DIR}/results/WING22_60min.json"
]

# #################### results
# ########## load results
objects, gains = loadResults(files=files, verbose=True)

# ########## get top 20 worst and top 20 best
topWorst, topBest = topWorstBest(
    top=20,
    objects=objects,
    gains=gains,
    verbose=True
)

# ########## pie plot with "Above 5,000", "Between 0 and 5,000",
# "Between -5,000 and 0", "Below -5,000" results
plotPies(
    objects=objects,
    gains=gains,
    border=5000,
    time_frame="60 min"
)
