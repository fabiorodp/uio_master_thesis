# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import os
from helper import loadResults, plotPies, topWorstBest

ROOT_DIR = os.path.abspath(os.curdir)

files = [
    f"{ROOT_DIR}/results/WINJ21_500000ticks.json",
    f"{ROOT_DIR}/results/WINM21_500000ticks.json",
    f"{ROOT_DIR}/results/WINQ21_500000ticks.json",
    f"{ROOT_DIR}/results/WINV21_500000ticks.json",
    f"{ROOT_DIR}/results/WINZ21_500000ticks.json",
    f"{ROOT_DIR}/results/WING22_500000ticks.json"
]

# #################### results
# ########## load results
objects, gains = loadResults(files=files, verbose=True)

# ########## get top 50 worst and top 50 best
topWorst, topBest = topWorstBest(
    top=50,
    objects=objects,
    gains=gains,
    verbose=True
)

# ########## pie plot with "Above 10,000", "Between 0 and 10,000",
# "Between -10,000 and 0", "Below -10,000" results
plotPies(
    objects=objects,
    gains=gains,
    border=10000,
    time_frame="500k ticks"
)
