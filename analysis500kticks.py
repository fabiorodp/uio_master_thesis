# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no
import pandas as pd

from helper import plotReturnTrajectories, plotMeanReturnTrajectory
from helper import run500times, optimal500, plotDist, plotBox
import pandas as pd
import numpy as np
import os

ROOT_DIR = os.path.abspath(os.curdir)

files = [
    f"{ROOT_DIR}/data/WINJ21/WINJ21_500000ticks.json",
    f"{ROOT_DIR}/data/WINM21/WINM21_500000ticks.json",
    f"{ROOT_DIR}/data/WINQ21/WINQ21_500000ticks.json",
    f"{ROOT_DIR}/data/WINV21/WINV21_500000ticks.json",
    f"{ROOT_DIR}/data/WINZ21/WINZ21_500000ticks.json",
    f"{ROOT_DIR}/data/WING22/WING22_500000ticks.json"
]

# #################### Discussion
params = ["GreedyGQ", 5, "sigmoid123", "minusMean", 0.1, 0.95, 0.1, 200]
objectsGreedyGQ = run500times(params, files)
optimalGreedyGQ = optimal500(objectsGreedyGQ)

plotReturnTrajectories(
    optimal=optimalGreedyGQ,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="500k--tick"
)

plotMeanReturnTrajectory(
    optimal=optimalGreedyGQ,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="500k--tick"
)

plotDist(
    optimal=optimalGreedyGQ,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="500k--tick"
)

plotBox(
    optimal=optimalGreedyGQ,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="500k--tick"
)

params = ["QLearn", 5, "hypTanh", "minusMean", 0.1, 0.95, 200]
objectsQLearn = run500times(params, files)
optimalQLearn = optimal500(objectsQLearn)

plotReturnTrajectories(
    optimal=optimalQLearn,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="500k--tick"
)

plotMeanReturnTrajectory(
    optimal=optimalQLearn,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="500k--tick"
)

plotDist(
    optimal=optimalQLearn,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="500k--tick"
)

plotBox(
    optimal=optimalQLearn,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="500k--tick"
)

params = ["SARSA", 5, "hypTanh", "minusMean", 0.1, 1, 0.1, "uniform01", 0]
objectsSARSA = run500times(params, files)
optimalSARSA = optimal500(objectsSARSA)

plotReturnTrajectories(
    optimal=optimalSARSA,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="500k--tick"
)

plotMeanReturnTrajectory(
    optimal=optimalSARSA,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="500k--tick"
)

plotDist(
    optimal=optimalSARSA,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="500k--tick"
)

plotBox(
    optimal=optimalSARSA,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="500k--tick"
)

# ########## saving for combined analysis
optimals = np.hstack(
    [
        optimalGreedyGQ["histRprime"][:, -1].reshape(-1, 1),
        optimalQLearn["histRprime"][:, -1].reshape(-1, 1),
        optimalSARSA["histRprime"][:, -1].reshape(-1, 1)
    ]
)

optimals = pd.DataFrame(
    optimals,
    columns=["GreedyGQ500k--tick", "QLearn500k--tick", "SARSA500k--tick"]
)

optimals.to_csv(
    f"{ROOT_DIR}/results/optimals500k--tick.csv",
    index_label=True
)
