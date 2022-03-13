# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
from algo1 import Agent
from env1 import Environment

fileName = "data/WING22/WING22_60min_OLHCV.csv"

params = {
    "n": [2],  # , 5, 10, 50],
    "basisFctType": ["sigmoid"],  # "hypTanh", "sigmoid", "relu"],
    "rewardType": ["mean"],  # "sum"],  # "shapeRatio" not working
    "eta": [0.1],  # 0.01, 0.001],
    "gamma": [1],  # 0.95, 1],
    "epsilon": [0.1],  # 0.2, 0.15, 0.10, 0.05],
    "initType": ["uniform01"],
    "lrScheduler": [False],  # False],
    "seed": [i for i in range(1, 6)]
}

save = {
    "params": [],
    "histTradePLs": [],
    "sumTradePLs": [],
    "deltas": [],
}

saved = {
    "params": [],
    "histTradePLs": [],
    "sumTradePLs": [],
    "meanSumTradePLs": []
}

for a in params["n"]:
    for b in params["basisFctType"]:
        for c in params["rewardType"]:
            for d in params["eta"]:
                for e in params["gamma"]:
                    for f in params["epsilon"]:
                        for g in params["initType"]:
                            for h in params["lrScheduler"]:
                                saved["params"].append((a, b, c, d, e, f, g, h))
                                histTradePLs = []
                                sumTradePLs = []

                                for i in params["seed"]:
                                    initInvest = 5600*5

                                    env = Environment(
                                        n=a,
                                        fileName=fileName,
                                        initInvest=initInvest,
                                        seed=i
                                    )

                                    agent = Agent(
                                        env=env,
                                        n=a,
                                        initInvest=initInvest,
                                        eta=d,
                                        gamma=e,
                                        epsilon=f,
                                        initType=g,
                                        rewardType=c,
                                        basisFctType=b,
                                        typeFeatureVector="block",
                                        lrScheduler=h,
                                        verbose=False,
                                        seed=i,
                                    )

                                    while env.terminal is not True:
                                        agent.run()

                                    save["params"].append((a, b, c, d, e, f, g, h, i))
                                    save["histTradePLs"].append(env.histTradePLs)
                                    save["sumTradePLs"].append(sum(env.histTradePLs))
                                    save["deltas"].append(agent.deltas)

                                    histTradePLs.append(env.histTradePLs)
                                    sumTradePLs.append(sum(env.histTradePLs))

                                saved["histTradePLs"].append(histTradePLs)
                                saved["sumTradePLs"].append(sumTradePLs)
                                saved["meanSumTradePLs"].append(np.mean(sumTradePLs))
