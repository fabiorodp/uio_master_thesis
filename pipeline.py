# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

from algorithm1 import Agent
from environment3b import Environment

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

"""params = {
    "n": [2, 5, 10, 15, 30, 60, 90, 120],
    "basisFctType": ["hypTanh123", "tanh", "sigmoid", "relu"],
    "rewardType": ["shapeRatio", "mean", "sum"],
    "eta": [0.5, 0.1, 0.05, 0.001, 0.0005, 0.0001, 0.00005, 0.000001],
    "gamma": [0.25, 0.5, 0.75, 0.90, 0.95, 0.975, 1],
    "epsilon": [0.25, 0.2, 0.15, 0.10, 0.05, 0.025, 0.01],
    "tradeRandEpsilon": [True, False],
    "initType": ["zeros", "uniform01"]
}"""

params = {
    "n": [2, 5],  # 10, 60, 90],
    "basisFctType": ["hypTanh123", "hypTanh", "sigmoid", "relu"],
    "rewardType": ["shapeRatio", "mean", "sum"],
    "eta": [0.1, 0.001, 0.0001],
    "gamma": [0.90, 0.95, 1],
    "epsilon": [0.2, 0.15, 0.10, 0.05],
    "tradeRandEpsilon": [True, False],
    "initType": ["zeros", "uniform01"]
}

save = {
    "params": [],
    "histR": [],
    "sumTradePL": [],
    "deltas": [],
    "cumPLs": []
}

fileName = "data/WING22/WING22_1min_OLHCV.csv"

for a in params["n"]:
    for b in params["basisFctType"]:
        for c in params["rewardType"]:
            for d in params["eta"]:
                for e in params["gamma"]:
                    for f in params["epsilon"]:
                        for g in params["tradeRandEpsilon"]:
                            for h in params["initType"]:

                                env = Environment(
                                    n=a,
                                    fileName=fileName
                                )

                                agent = Agent(
                                    env=env,
                                    n=a,
                                    eta=d,
                                    gamma=e,
                                    epsilon=f,
                                    initType=h,
                                    rewardType=c,
                                    basisFctType=b,
                                    typeFeatureVector="block",
                                    tradeRandEpsilon=g,
                                    verbose=False,
                                    seed=20,
                                )

                                while env.terminal is not True:
                                    agent.run()

                                cumReturn, taus = [], []
                                for i in range(len(agent.memory)):
                                    if (agent.memory["tradeStatus"][i] == 0) and (agent.memory["A"][i] == -1):
                                        cumReturn.append(agent.memory["tradePL"][i-1])
                                        taus.append(agent.memory["tau"][i - 1])

                                    elif (agent.memory["tradeStatus"][i] == 0) and (agent.memory["A"][i] == 1):
                                        cumReturn.append(agent.memory["tradePL"][i-1])
                                        taus.append(agent.memory["tau"][i - 1])

                                save["params"].append((a, b, c, d, e, f, g, h))
                                save["histR"].append(env.histR)
                                save["sumTradePL"].append(sum(cumReturn))
                                save["deltas"].append(agent.deltas)
                                save["cumPLs"].append(cumReturn)
