# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helper import readPythonObjectFromFile
from helper import plotReturnTrajectories, plotMeanReturnTrajectory


files = [
    "results/WINJ21_500000ticks.json",
    "results/WINM21_500000ticks.json",
    # "results/WINQ21_250000ticks.json",
    # "results/WINV21_250000ticks.json",
    # "results/WINZ21_250000ticks.json",
    # "results/WING22_250000ticks.json"
]

WINJ21 = readPythonObjectFromFile(
    path=files[0],
    openingType="json"
)

WINM21 = readPythonObjectFromFile(
    path=files[1],
    openingType="json"
)

# ########## pick the best combination of hyper-parameters
gains = np.array(WINJ21["meanSumTradePLs"]) + \
        np.array(WINM21["meanSumTradePLs"])

b = np.argsort(gains)
c = np.sort(gains)
best = -1
argBest = int(b[best])

# ########## get the optimal hyper-parameters and its results
optimal = {
    "params": WINJ21["params"][argBest],
    "arg": int(argBest),
    "histRprime": np.array(WINJ21["histRprime"][argBest]),
    "meanPL": c[best]
}

objects = (
    WINJ21,
    WINM21
)

# ########## merge the histRprime trajectories
for obt in objects[1:]:
    lastCol = optimal["histRprime"][:, -1][:, None]
    arr1 = np.array(obt["histRprime"][optimal["arg"]]) - 28000 + lastCol
    arr1 = np.hstack([optimal["histRprime"], arr1])
    optimal["histRprime"] = arr1

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
df = pd.Series(optimal["histRprime"][:, -1])
df.describe()

optimal["histRprime"].std(axis=0)
