# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

from helper import plotReturnTrajectories, plotMeanReturnTrajectory
from helper import loadResults, plotPies, topWorstBest, getOptimal
from helper import run500times, optimal500
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

files = [
    "results/WINJ21/WINJ21_500000ticks.json",
    "results/WINM21/WINM21_500000ticks.json",
    "results/WINQ21/WINQ21_500000ticks.json",
    "results/WINV21/WINV21_500000ticks.json",
    "results/WINZ21/WINZ21_500000ticks.json",
    "results/WING22/WING22_500000ticks.json"
]

# #################### Discussion
params = ["GreedyGQ", 5, "sigmoid123", "minusMean", 0.1, 0.95, 0.1, 200]
objectsGreedyGQ = run500times(params)
optimalGreedyGQ = optimal500(objectsGreedyGQ)
plotReturnTrajectories(optimalGreedyGQ)
plotMeanReturnTrajectory(optimalGreedyGQ)

params = ["QLearn", 5, "hypTanh", "minusMean", 0.1, 0.95, 200]
objectsQLearn = run500times(params)
optimalQLearn = optimal500(objectsQLearn)
plotReturnTrajectories(optimalQLearn)
plotMeanReturnTrajectory(optimalQLearn)

params = ["SARSA", 5, "hypTanh", "minusMean", 0.1, 1, 0.1, "uniform01", 0]
objectsSARSA = run500times(params)
optimalSARSA = optimal500(objectsSARSA)
plotReturnTrajectories(optimalSARSA)
plotMeanReturnTrajectory(optimalSARSA)

# ########## hist plot distribution of the final returns
plt.hist(optimalGreedyGQ["histRprime"][:, -1], density=True)
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.show()

plt.hist(optimalQLearn["histRprime"][:, -1], density=True)
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.show()

plt.hist(optimalSARSA["histRprime"][:, -1], density=True)
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.show()

# ########## box plot distribution of the final returns
sns.boxplot(optimalGreedyGQ["histRprime"][:, -1])
plt.show()

sns.boxplot(optimalQLearn["histRprime"][:, -1])
plt.show()

sns.boxplot(optimalSARSA["histRprime"][:, -1])
plt.show()
