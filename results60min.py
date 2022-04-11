# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

from helper import loadResults, plotPies, topWorstBest

files = [
    "results/WINJ21/WINJ21_60min.json",
    "results/WINM21/WINM21_60min.json",
    "results/WINQ21/WINQ21_60min.json",
    "results/WINV21/WINV21_60min.json",
    "results/WINZ21/WINZ21_60min.json",
    "results/WING22/WING22_60min.json"
]

# #################### results
# ########## load results
objects, gains = loadResults(files)

# ########## get top 20 worst and top 20 best
topWorst, topBest = topWorstBest(
    top=20,
    objects=objects,
    gains=gains
)

# ########## pie plot with "Above 5,000", "Between 0 and 5,000",
# "Between -5,000 and 0", "Below -5,000" results
plotPies(
    objects=objects,
    gains=gains,
    border=5000,
    time_frame="60 min"
)
