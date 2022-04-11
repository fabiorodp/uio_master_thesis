# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

from helper import loadResults, plotPies, topWorstBest

files = [
    "results/WINJ21/WINJ21_500000ticks.json",
    "results/WINM21/WINM21_500000ticks.json",
    "results/WINQ21/WINQ21_500000ticks.json",
    "results/WINV21/WINV21_500000ticks.json",
    "results/WINZ21/WINZ21_500000ticks.json",
    "results/WING22/WING22_500000ticks.json"
]

# #################### results
# ########## load results
objects, gains = loadResults(files)

# ########## get top 20 worst and top 20 best
topWorst, topBest = topWorstBest(
    top=50,
    objects=objects,
    gains=gains
)

# ########## pie plot with "Above 10,000", "Between 0 and 10,000",
# "Between -10,000 and 0", "Below -10,000" results
plotPies(
    objects=objects,
    gains=gains,
    border=10000,
    time_frame="500k ticks"
)
