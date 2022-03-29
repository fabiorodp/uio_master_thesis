# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helper import readPythonObjectFromFile
from helper import plotReturnTrajectories, plotMeanReturnTrajectory
from helper import loadResults, topWorstBestAndOptimal, getOptimal

files = [
    "results/WINJ21_500000ticks.json",
    "results/WINM21_500000ticks.json",
    # "results/WINQ21_500000ticks.json",
    # "results/WINV21_500000ticks.json",
    # "results/WINZ21_500000ticks.json",
    # "results/WING22_500000ticks.json"
]

objects, gains = loadResults(files)

topWorst, topBest = topWorstBestAndOptimal(
    top=20,
    objects=objects,
    gains=gains
)

optimal = getOptimal(
    objects=objects,
    gains=gains,
    optimalID=-1
)

# ########## line plot trajectories
plotReturnTrajectories(optimal)

# ########## line plot mean trajectory
plotMeanReturnTrajectory(optimal)

# ########## hist plot distribution of the final returns
plt.hist(optimal["histRprime"][:, -1], density=True)
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.show()

# ########## pie plot counting positive and negative periods
"""myexplode = [0.2]
plt.pie(
    np.sum(optimal["histRprime"][:, -1] >= 28000),
    labels=[True, False],
    explode=myexplode,
    shadow=True
)
plt.legend(title="Gains of the trades:")
plt.show()"""

# ########## box plot distribution of the final returns
sns.boxplot(optimal["histRprime"][:, -1])
plt.show()

# ########## descriptive statistics
pd.Series(optimal["histRprime"][:, -1]).describe()
