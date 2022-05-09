# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

from helper import plotReturnTrajectories, plotMeanReturnTrajectory
from helper import run500times, optimal500, plotDist, plotBox
import os

ROOT_DIR = os.path.abspath(os.curdir)

files = [
    f"{ROOT_DIR}/data/WINJ21/WINJ21_60min.json",
    f"{ROOT_DIR}/data/WINM21/WINM21_60min.json",
    f"{ROOT_DIR}/data/WINQ21/WINQ21_60min.json",
    f"{ROOT_DIR}/data/WINV21/WINV21_60min.json",
    f"{ROOT_DIR}/data/WINZ21/WINZ21_60min.json",
    f"{ROOT_DIR}/data/WING22/WING22_60min.json"
]

# #################### Discussion
params = ["GreedyGQ", 5, "sigmoid", "minusMean", 0.01, 0.95, 0.1, 200]
objectsGreedyGQ = run500times(params, files)
optimalGreedyGQ = optimal500(objectsGreedyGQ)

plotReturnTrajectories(
    optimal=optimalGreedyGQ,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="60 min"
)

plotMeanReturnTrajectory(
    optimal=optimalGreedyGQ,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="60 min"
)

plotDist(
    optimal=optimalGreedyGQ,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="60 min"
)

plotBox(
    optimal=optimalGreedyGQ,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="60 min"
)

params = ["QLearn", 5, "sigmoid", "minusMean", 0.01, 0.95, 0]
objectsQLearn = run500times(params, files)
optimalQLearn = optimal500(objectsQLearn)

plotReturnTrajectories(
    optimal=optimalQLearn,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="60 min"
)

plotMeanReturnTrajectory(
    optimal=optimalQLearn,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="60 min"
)

plotDist(
    optimal=optimalQLearn,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="60 min"
)

plotBox(
    optimal=optimalQLearn,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="60 min"
)

params = ["SARSA", 5, "hypTanh", "minusMean", 0.01, 1, 0.1, "zeros", 200]
objectsSARSA = run500times(params, files)
optimalSARSA = optimal500(objectsSARSA)

plotReturnTrajectories(
    optimal=optimalSARSA,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="60 min"
)

plotMeanReturnTrajectory(
    optimal=optimalSARSA,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="60 min"
)

plotDist(
    optimal=optimalSARSA,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="60 min"
)

plotBox(
    optimal=optimalSARSA,
    initInvest=28000,
    algoType=params[0],
    showPlot=True,
    timeFramed="60 min"
)
