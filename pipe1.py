# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
from algo1 import Agent
from env1 import Environment


params = {
    "n": [2, 5, 10, 50],
    "basisFctType": ["hypTanh123", "hypTanh", "sigmoid", "relu"],
    "rewardType": ["shapeRatio", "mean", "sum"],
    "eta": [0.1, 0.01, 0.001],
    "gamma": [0.90, 0.95, 1],
    "epsilon": [-1, 0.2, 0.15, 0.10, 0.05],
    "initType": ["uniform01"],
    "lrScheduler": [True, False],
    "seed": [i for i in range(1, 21)]
}

save = {
    "params": [],
    "histTradePLs": [],
    "deltas": [],
}

save1 = {
    "params": [],
    "meanPL": []
}

fileName = "data/WING22/WING22_60min_OLHCV.csv"


for a in params["n"]:
    for b in params["basisFctType"]:
        for c in params["rewardType"]:
            for d in params["eta"]:
                for e in params["gamma"]:
                    for f in params["epsilon"]:
                        for g in params["initType"]:
                            save1["params"].append((a, b, c, d, e, f, g))
                            meanPL = []

                            for h in params["seed"]:
                                env = Environment(
                                    n=a,
                                    fileName=fileName,
                                    initInvest=5600*5
                                )

                                agent = Agent(
                                    env=env,
                                    n=a,
                                    initInvest=5600*5,
                                    eta=d,
                                    gamma=e,
                                    epsilon=f,
                                    initType=g,
                                    rewardType=c,
                                    basisFctType=b,
                                    typeFeatureVector="block",
                                    lrScheduler=True,
                                    verbose=False,
                                    seed=h,
                                )

                                while env.terminal is not True:
                                    agent.run()

                                save["params"].append((a, b, c, d, e, f, g, h))
                                save["histTradePLs"].append(env.histTradePLs)
                                save["deltas"].append(agent.deltas)
                                meanPL.append(sum(env.histTradePLs))

                            save1["meanPL"].append(np.mean(meanPL))
