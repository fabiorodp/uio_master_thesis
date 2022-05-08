# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import os
import re
import numpy as np
from tqdm import tqdm
from environment import Environment
from algorithms import SARSA, QLearn, GreedyGQ
from helper import savePythonObject


def pipeline(filePath, initInvest=5600 * 5, params=None, outFolder=None,
             verbose=True):
    """
    This is a function to run the pipeline with hyper-parameters' search.

    Parameters:
    ===================

    :param filePath: str:       Input the entire path for the parsed data.
                                - For example:
                                "./data/WING22/WING22_500000ticks.csv"

    :param initInvest: float:   The value of the initial investment capital.
                                - For example: 5600 * 5

    :param params: dict:        Dictionary with params to search.
                                - Attention to the dictionary format,
                                which must be:

                                params = {
                                    "rlType": [],
                                    "n": [],
                                    "basisFctType": [],
                                    "rewardType": [],
                                    "eta": [],
                                    "zeta": [],
                                    "gamma": [],
                                    "epsilon": [],
                                    "initType": [],
                                    "lrScheduler": [],
                                    "seed": []
                                }

    :param outFolder: str:      Path to save results.
                                - If None, the path is
                                ROOT_DIR+'/'+/results/'.

    :param verbose: bool:       Print all steps of this module.

    Return:
    ===================
        This function does not return anything, but only saves a json
        file with the results.
    """
    if params is None:
        params = {
            "rlType": ["SARSA", "QLearn", "GreedyGQ"],
            "n": [5, 25, 50],
            "basisFctType": ["sigmoid", "sigmoid123", "hypTanh"],
            "rewardType": ["minusMean", "immediate", "mean"],
            "eta": [0.1, 0.01],
            "zeta": [0.1, 0.01],
            "gamma": [1, 0.95],
            "epsilon": [0.15, 0.1],
            "initType": ["uniform01", "zeros"],
            "lrScheduler": [0, 200],
            "seed": [i for i in range(1, 51)]
        }

    saved = {
        "params": [],
        "TDErrors": [],
        "histRprime": [],
        "sumTradePLs": [],
        "meanSumTradePLs": []
    }

    for ag in tqdm(params["rlType"], desc="Loading pipeline..."):

        if ag == "SARSA":
            for a in tqdm(params["n"], desc="Loading n for SARSA..."):
                for b in params["basisFctType"]:
                    for c in params["rewardType"]:
                        for d in params["eta"]:
                            for e in params["gamma"]:
                                for f in params["epsilon"]:
                                    for g in params["initType"]:
                                        for h in params["lrScheduler"]:

                                            saved["params"].append(
                                                (ag, a, b, c, d, e, f, g, h))

                                            TDErrors = []
                                            sumTradePLs = []
                                            histRprime = []

                                            for i in params["seed"]:

                                                env = Environment(
                                                    n=a,
                                                    fileName=filePath,
                                                    initInvest=initInvest,
                                                    seed=i
                                                )

                                                agent = SARSA(
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

                                                TDErrors.append(agent.TDErrors)

                                                sumTradePLs.append(
                                                    sum(env.histTradePLs))

                                                histRprime.append(
                                                    env.histRprime)

                                            saved["TDErrors"].append(TDErrors)

                                            saved["sumTradePLs"].append(
                                                sumTradePLs)

                                            saved["histRprime"].append(
                                                histRprime)

                                            saved["meanSumTradePLs"].append(
                                                np.mean(sumTradePLs))

        elif ag == "QLearn":
            for a in tqdm(params["n"], desc="Loading n for QLearn..."):
                for b in params["basisFctType"]:
                    for c in params["rewardType"]:
                        for d in params["eta"]:
                            for e in params["gamma"]:
                                for h in params["lrScheduler"]:

                                    saved["params"].append(
                                        (ag, a, b, c, d, e, h))

                                    TDErrors = []
                                    sumTradePLs = []
                                    histRprime = []

                                    for i in params["seed"]:

                                        env = Environment(
                                            n=a,
                                            fileName=filePath,
                                            initInvest=initInvest,
                                            seed=i
                                        )

                                        agent = QLearn(
                                            env=env,
                                            n=a,
                                            initInvest=initInvest,
                                            eta=d,
                                            gamma=e,
                                            initType="uniform01",
                                            rewardType=c,
                                            basisFctType=b,
                                            typeFeatureVector="block",
                                            lrScheduler=h,
                                            verbose=False,
                                            seed=i,
                                        )

                                        while env.terminal is not True:
                                            agent.run()

                                        TDErrors.append(agent.TDErrors)

                                        sumTradePLs.append(
                                            sum(env.histTradePLs))

                                        histRprime.append(env.histRprime)

                                    saved["TDErrors"].append(TDErrors)

                                    saved["sumTradePLs"].append(
                                        sumTradePLs)

                                    saved["histRprime"].append(histRprime)

                                    saved["meanSumTradePLs"].append(
                                        np.mean(sumTradePLs))

        elif ag == "GreedyGQ":
            for a in tqdm(params["n"], desc="Loading n for GreedyGQ..."):
                for b in params["basisFctType"]:
                    for c in params["rewardType"]:
                        for d in params["eta"]:
                            for e in params["gamma"]:
                                for f in params["zeta"]:
                                    for h in params["lrScheduler"]:

                                        saved["params"].append(
                                            (ag, a, b, c, d, e, f, h))

                                        TDErrors = []
                                        sumTradePLs = []
                                        histRprime = []

                                        for i in params["seed"]:

                                            env = Environment(
                                                n=a,
                                                fileName=filePath,
                                                initInvest=initInvest,
                                                seed=i
                                            )

                                            agent = GreedyGQ(
                                                env=env,
                                                n=a,
                                                initInvest=initInvest,
                                                eta=d,
                                                gamma=e,
                                                zeta=f,
                                                initType="uniform01",
                                                rewardType=c,
                                                basisFctType=b,
                                                typeFeatureVector="block",
                                                lrScheduler=h,
                                                verbose=False,
                                                seed=i,
                                            )

                                            while env.terminal is not True:
                                                agent.run()

                                            TDErrors.append(agent.TDErrors)

                                            sumTradePLs.append(
                                                sum(env.histTradePLs))

                                            histRprime.append(env.histRprime)

                                        saved["TDErrors"].append(TDErrors)

                                        saved["sumTradePLs"].append(
                                            sumTradePLs)

                                        saved["histRprime"].append(histRprime)

                                        saved["meanSumTradePLs"].append(
                                            np.mean(sumTradePLs))

        print(f"Complete {ag}.")

    print(f"Complete all.")

    # ########## saving results
    ROOT_DIR = os.path.abspath(os.curdir)

    if outFolder is None:
        outFolder = ROOT_DIR+'/'+'/results/'

        if not os.path.isdir(outFolder):
            os.makedirs(outFolder)

            if verbose:
                print(f"LOG: Directory {outFolder} created.")

    else:
        if not os.path.isdir(outFolder):
            raise ValueError(f"ERROR: The full path for outFolder="
                             f"'{outFolder}' parameter was not found.")

    fname = re.search(r"\w+/(\w+).csv", filePath).group(1)
    savePythonObject(
        pathAndFileName=f"{outFolder}{fname}",
        pythonObject=saved,
        savingType="json"
    )
