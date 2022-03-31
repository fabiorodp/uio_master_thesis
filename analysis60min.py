# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

from helper import plotReturnTrajectories, plotMeanReturnTrajectory
from helper import loadResults, plotPies, topWorstBest, getOptimal
from helper import run500times, optimal500
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

files = [
    "results/WINJ21_60min.json",
    "results/WINM21_60min.json",
    "results/WINQ21_60min.json",
    "results/WINV21_60min.json",
    "results/WINZ21_60min.json",
    "results/WING22_60min.json"
]

# ########## load results
objects, gains = loadResults(files)

# ########## pie plot counting positive and negative periods
plotPies(objects, gains)

# ########## get top 20 worst and top 20 best
topWorst, topBest = topWorstBest(
    top=20,
    objects=objects,
    gains=gains
)

# ########## get optimal for GreedyGQ (-1), QLearn (-2) and SARSA (-17)
optimalGreedyGQ = getOptimal(
    objects=objects,
    gains=gains,
    optimalID=-1
)

optimalQLearn = getOptimal(
    objects=objects,
    gains=gains,
    optimalID=-2
)

optimalSARSA = getOptimal(
    objects=objects,
    gains=gains,
    optimalID=-17
)

# ########## save memory
del files, gains, objects

# ########## run 500 seeds with the optimal parameters
objectsGreedyGQ = run500times(optimalGreedyGQ["params"])
objectsQLearn = run500times(optimalQLearn["params"])
objectsSARSA = run500times(optimalSARSA["params"])

# ########## 500 runs optimal
optimalGreedyGQ = optimal500(objectsGreedyGQ)
optimalQLearn = optimal500(objectsQLearn)
optimalSARSA = optimal500(objectsSARSA)

# ########## line plot trajectories
plotReturnTrajectories(optimalGreedyGQ)
plotReturnTrajectories(optimalQLearn)
plotReturnTrajectories(optimalSARSA)

# ########## line plot mean trajectory
plotMeanReturnTrajectory(optimalGreedyGQ)
plotMeanReturnTrajectory(optimalQLearn)
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
