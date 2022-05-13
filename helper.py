# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from environment import Environment
from algorithms import SARSA, QLearn, GreedyGQ


def getAnAsset(ticker='WINJ21', in_folder=None, out_folder=None,
               verbose=True):
    """
    This is a function to extract all negotiations of a ticker.

    Parameters:
    ===================
    :param ticker: str:         The financial security symbol.

    :param in_folder: str:      The full path where the data files are stored.
                                - If None, the full path will be
                                ROOT_DIR+'/'+'data/'+ticker+'/raw/'

    :param out_folder: str:     The full Path where the CSV data file
                                will be stored.

    :param verbose: bool:       Print all steps of this module.

    Return:
    ===================
        It does not return anything, only saves the extracted data for the
        given ticker.
    """
    # ########## Managing directories' paths
    ROOT_DIR = os.path.abspath(os.curdir)

    if in_folder is None:
        in_folder = ROOT_DIR+'/'+'data/'+ticker+'/raw/'

        if not os.path.isdir(in_folder):
            raise ValueError(f"ERROR: The full path for in_folder="
                             f"'{in_folder}' parameter was not found.")

        if len(os.listdir(in_folder)) == 0:
            raise ValueError(f"ERROR: Directory in_folder={in_folder}"
                             f" is empty.")
    else:
        if not os.path.isdir(in_folder):
            raise ValueError(f"ERROR: The full path for in_folder="
                             f"'{in_folder}' parameter was not found.")

    if out_folder is None:
        out_folder = ROOT_DIR+'/'+'data/'+ticker+'/extracted/'

        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
            if verbose:
                print(f"LOG: Directory {out_folder} created.")
    else:
        if not os.path.isdir(out_folder):
            raise ValueError(f"ERROR: The full path for out_folder="
                             f"'{out_folder}' parameter was not found.")
    # ##########

    # ########## Iterating over a directory
    for filename in os.listdir(in_folder):

        if verbose:
            print(f"Extracting {ticker} from {filename}...")

        if filename.endswith(".zip") and \
                filename.startswith('TradeIntraday_'):

            # importing data-set file
            data = pd.read_csv(
                in_folder + filename,
                compression='zip',
                sep=';',
                header=0,
                dtype=str
            )

            # removing trades that were not for the selected ticker
            drop_idxs = data['TckrSymb'][data['TckrSymb'] != ticker].index
            data.drop(drop_idxs, axis=0, inplace=True)

            # dropping row indexes with 'TradgSsnId' == 2
            # because they are cancelled trades
            drop_idxs = data['TradgSsnId'][data['TradgSsnId'] == 2].index
            data.drop(drop_idxs, axis=0, inplace=True)

            # dropping unnecessary columns
            data.drop(['TckrSymb', 'RptDt', 'UpdActn', 'TradId',
                       'TradgSsnId'], axis=1, inplace=True)

            # fixing data and time
            data["DateTime"] = data['TradDt'] + ' ' + data['NtryTm']

            # dropping unnecessary columns
            data.drop(['NtryTm', 'TradDt'], axis=1, inplace=True)

            # converting data type
            data["DateTime"] = pd.to_datetime(
                data["DateTime"],
                format='%Y-%m-%d %H%M%f'
            )

            # replacing "," to "." in price
            data.columns = ["Price", "Volume", "DateTime"]
            data["Price"] = data["Price"].str.replace(',', '.')

            # fixing dtypes
            data["Price"] = data["Price"].astype(np.float64)
            data["Volume"] = data["Volume"].astype(np.int64)

            # dropping old index
            data.reset_index(inplace=True, drop='index')

            # creating csv data file
            fname = re.search(
                r'TradeIntraday_(\d\d\d\d\d\d\d\d)_1.zip',
                filename
            ).group(1)

            fname = ticker + "_" + fname + ".csv"

            data.to_csv(
                f"{out_folder+'/'+fname}",
                sep=';',
                index_label=False
            )
    # ##########
    savedFiles = []
    for i in os.listdir(out_folder):
        if i.startswith(ticker) and i.endswith(".csv"):
            savedFiles.append(i)

    if verbose:
        print(f"The list of files {savedFiles} was saved "
              f"in {out_folder}.")
        print("Extraction done.")


def parseIntoTimeBars(ticker='WINJ21', candles_periodicity='1D',
                      in_folder=None, out_folder=None,
                      verbose=True):
    """
    This is a function to create time candles (framed intervals) data
    based on a given ticker/contract/asset.

    Parameters:
    ===================
    :param ticker: str:                 The financial instrument (ticker).
                                        Default: 'WINJ21'.

    :param candles_periodicity: str:    Periodicity of the candle. Default
                                        '1D' that means 1 day.
                                        Options: 'xmin' where x is the
                                        number of minutes.

    :param in_folder: str:              The folder where the data file
                                        containing all negotiations of
                                        the ticker is stored.
                                        - If None, the full path will be
                                        ROOT_DIR+'/'+'data/'+ticker+'/extracted/'

    :param out_folder: str:             The Path where the CSV data file
                                        will be stored.
                                        - If None, the full path will be
                                        ROOT_DIR+'/'+'data/'+ticker+'/'

    :param verbose: bool:               Print all steps of this module.

    Return:
    ===================
    It saves the parsed data for the given ticker.

    :returns data: pd.DataFrame:        DataFrame containing the OLHCV
                                        data for the given ticker and
                                        periodicity.
    """
    # ########## Managing directories' paths
    ROOT_DIR = os.path.abspath(os.curdir)

    if in_folder is None:
        in_folder = ROOT_DIR + '/' + 'data/' + ticker + '/extracted/'

        if not os.path.isdir(in_folder):
            raise ValueError(f"ERROR: The full path for in_folder="
                             f"'{in_folder}' parameter was not found.")

        if len(os.listdir(in_folder)) == 0:
            raise ValueError(f"ERROR: Directory in_folder={in_folder}"
                             f" is empty.")
    else:
        if not os.path.isdir(in_folder):
            raise ValueError(f"ERROR: The full path for in_folder="
                             f"'{in_folder}' parameter was not found.")

    if out_folder is None:
        out_folder = ROOT_DIR + '/' + 'data/' + ticker + '/'

    if not os.path.isdir(out_folder):
        raise ValueError(f"ERROR: The full path for out_folder="
                         f"'{out_folder}' parameter was not found.")
    # ##########

    data = pd.DataFrame()
    for file in os.listdir(in_folder):

        if verbose:
            print(f"Parsing {ticker} from {file}...")

        if file.endswith(".csv") and file.startswith(f'{ticker}'):
            df = pd.read_csv(in_folder + file, sep=';')
            df.set_index(pd.DatetimeIndex(df['DateTime']), inplace=True)
            time_candle = df.Price.resample(candles_periodicity).ohlc()
            grouped = df.groupby(pd.Grouper(freq=candles_periodicity)).sum()
            time_candle['volume'] = grouped.Volume
            data = pd.concat([data, time_candle])

    data.sort_index(inplace=True)
    data.dropna(inplace=True)

    # creating csv data file
    fname = ticker + "_" + candles_periodicity + ".csv"

    data.to_csv(
        f"{out_folder + '/' + fname}",
        sep=';',
        index_label=False
    )

    savedFiles = []
    for i in os.listdir(out_folder):
        if i.startswith(ticker) \
                and i.endswith(f"{candles_periodicity +'.csv'}"):
            savedFiles.append(i)

    if verbose:
        print(f"The list of files {savedFiles} was saved "
              f"in {out_folder}.")
        print("Parsing done.")

    return data


def parseIntoTickBars(ticker='WING22', numTicks=500000,
                      in_folder=None, out_folder=None,
                      verbose=True):
    """
    This is a function to create tick candles (framed intervals) data
    based on a given ticker/contract/asset.

    Parameters:
    ===================
    :param ticker: str:             The financial instrument (ticker).
                                    Default: 'WINJ21'.

    :param numTicks: str:           Periodicity of the candle. Default
                                    '1D' that means 1 day.
                                    Options: 'xmin' where x is the
                                    number of minutes.

    :param in_folder: str:          The folder where the data file
                                    containing all negotiations of
                                    the ticker is stored.
                                    - If None, the full path will be
                                    ROOT_DIR+'/'+'data/'+ticker+'/extracted/'

    :param out_folder: str:         The Path where the CSV data file
                                    will be stored.
                                    - If None, the full path will be
                                    ROOT_DIR+'/'+'data/'+ticker+'/'

    :param verbose: bool:           Print all steps of this module.

    Return:
    ===================
    It saves the parsed data for the given ticker.

    :returns data: pd.DataFrame:        DataFrame containing the OLHCV
                                        data for the given ticker and
                                        periodicity.
    """
    # ########## Managing directories' paths
    ROOT_DIR = os.path.abspath(os.curdir)

    if in_folder is None:
        in_folder = ROOT_DIR + '/' + 'data/' + ticker + '/extracted/'

        if not os.path.isdir(in_folder):
            raise ValueError(f"ERROR: The full path for in_folder="
                             f"'{in_folder}' parameter was not found.")

        if len(os.listdir(in_folder)) == 0:
            raise ValueError(f"ERROR: Directory in_folder={in_folder}"
                             f" is empty.")
    else:
        if not os.path.isdir(in_folder):
            raise ValueError(f"ERROR: The full path for in_folder="
                             f"'{in_folder}' parameter was not found.")

    if out_folder is None:
        out_folder = ROOT_DIR + '/' + 'data/' + ticker + '/'

    if not os.path.isdir(out_folder):
        raise ValueError(f"ERROR: The full path for out_folder="
                         f"'{out_folder}' parameter was not found.")
    # ##########

    data = pd.DataFrame()
    for file in os.listdir(in_folder):

        if verbose:
            print(f"Parsing {ticker} from {file}...")

        if file.endswith(".csv") and file.startswith(f'{ticker}'):
            df = pd.read_csv(in_folder + file, sep=';')
            df.set_index(pd.DatetimeIndex(df['DateTime']), inplace=True)
            data = pd.concat([data, df])

    # get infos
    columns = ["DateTime", "open", "high", "low", "close", "volume"]
    diff = len(data) // numTicks
    mod = len(data) % numTicks
    dfFinal = []

    for i in range(diff):
        # print(f"{i*numTicks}:{i*numTicks+numTicks}")
        Open = data.iloc[i * numTicks:i * numTicks + numTicks, 0][0]
        High = np.max(
            data.iloc[i * numTicks:i * numTicks + numTicks, 0].values)
        Low = np.min(
            data.iloc[i * numTicks:i * numTicks + numTicks, 0].values)
        Close = data.iloc[i * numTicks:i * numTicks + numTicks, 0][-1]
        Volume = np.sum(
            data.iloc[i * numTicks:i * numTicks + numTicks, 1].values)
        DateTime = data.iloc[i * numTicks:i * numTicks + numTicks, 2][-1]
        DateTime = datetime.strptime(DateTime, '%Y-%m-%d %H:%M:%S.%f')
        dfFinal.append([DateTime, Open, High, Low, Close, Volume])

    if mod != 0:
        # print(f"{-(len(df) % numTicks)}:")
        Open = data.iloc[-mod:, 0][0]
        High = np.max(data.iloc[-mod:, 0].values)
        Low = np.min(data.iloc[-mod:, 0].values)
        Close = data.iloc[-mod:, 0][-1]
        Volume = np.sum(data.iloc[-mod:, 1].values)
        DateTime = data.iloc[-mod:, 2][-1]
        DateTime = datetime.strptime(DateTime, '%Y-%m-%d %H:%M:%S.%f')
        dfFinal.append([DateTime, Open, High, Low, Close, Volume])

    dfFinal = pd.DataFrame(dfFinal, columns=columns)
    dfFinal.set_index(pd.DatetimeIndex(
        dfFinal['DateTime']), inplace=True)
    dfFinal.drop(['DateTime'], axis=1, inplace=True)

    # creating csv data file
    fname = ticker + "_" + str(numTicks) + "ticks.csv"

    dfFinal.to_csv(
        path_or_buf=f"{out_folder + '/' + fname}",
        sep=';',
        index_label=False
    )

    savedFiles = []
    for i in os.listdir(out_folder):
        if i.startswith(ticker) \
                and i.endswith(f"{str(numTicks) +'ticks.csv'}"):
            savedFiles.append(i)

    if verbose:
        print(f"The list of files {savedFiles} was saved "
              f"in {out_folder}.")
        print("Parsing done.")

    return dfFinal


def loadResults(files: list, verbose: bool = True):
    """
    Module to load results from files.

    Parameters:
    ===================
    :param files: list:             Python's list containing all results files.

    :param verbose: bool:           Print all steps of this module.

    Return:
    ===================
    :returns objects: list:         Python's object with the saved results.

    :returns gains: np.ndarray:     With the mean gains of each individual
                                    result.
    """
    objects, gains = [], None

    for idx, file in enumerate(files):

        if verbose:
            print(f"Reading {file}...")

        temp = readPythonObjectFromFile(
            path=file,
            openingType="json"
        )

        objects.append(temp)

        if idx == 0:
            gains = np.array(temp["meanSumTradePLs"])
        else:
            gains += np.array(temp["meanSumTradePLs"])

    if verbose:
        print(f"Loading is done.")

    return objects, gains


def topWorstBest(top, objects, gains, verbose: bool = True):
    """
    Module to filter the top worst/best results.

    Parameters:
    ===================
    :param top: int:                Input the wanted number of top occurrences.

    :param objects: list:           Input the list containing the saved
                                    results.

    :param gains: np.ndarray:       The mean gains of each individual result.

    :param verbose: bool:           Print all steps of this module.

    Return:
    ===================
    :returns objects: list:     Python's object with the results.

    :returns gains: ndarray:    With the mean gains of each individual result.
    """
    # ########## pick the best combination of hyper-parameters
    b = np.argsort(gains)
    c = np.sort(gains)

    # ########## top 10 worst and top 10 best parameters and scores
    argTopWorst = [a for a in b[:top]]
    topWorst = {
        "args": argTopWorst,
        "params": [[a for a in objects[0]["params"][i]]
                   for i in argTopWorst],
        "scores": [a for a in c[:top]]
    }

    if verbose:
        print(f"Top {top} worst scores: {topWorst['scores']}")
        print(f"Top {top} worst params: {topWorst['params']}")

    argTopBest = [a for a in b[-top:]]
    topBest = {
        "args": argTopBest,
        "params": [[a for a in objects[0]["params"][i]]
                   for i in argTopBest],
        "scores": [a for a in c[-top:]]
    }

    if verbose:
        print(f"Top {top} best scores: {topBest['scores']}")
        print(f"Top {top} best params: {topBest['params']}")

    return topWorst, topBest


def plotPies(objects, gains, border=5000, time_frame="60 min"):
    """
    Function to show pie plots of results.

    Parameters:
    ===================
    :param objects: list:           Input the list containing the saved
                                    results.

    :param gains: np.ndarray:       The mean gains of each individual result.

    :param border: int:             Value used as a pre-defined proportion
                                    for each pie's division.

    :param time_frame: str:         String with the type of the framed
                                    intervals.

    Return:
    ===================
        This function does not return any object, but only shows the pie plots.
    """
    gainsGreedyGQ, gainsQLearn, gainsSARSA = [], [], []
    for idx, p in enumerate(objects[0]["params"]):
        if p[0] == "GreedyGQ":
            gainsGreedyGQ.append(gains[idx])
        elif p[0] == "QLearn":
            gainsQLearn.append(gains[idx])
        elif p[0] == "SARSA":
            gainsSARSA.append(gains[idx])

    gs = [
        gains.tolist(),
        gainsGreedyGQ,
        gainsQLearn,
        gainsSARSA
    ]

    for idx, g in enumerate(gs):
        logic = [
            np.sum(np.array(g) > border),
            np.sum((np.array(g) >= 0) & (np.array(g) <= border)),
            np.sum((np.array(g) >= -border) & (np.array(g) < 0)),
            np.sum(np.array(g) < -border)
        ]

        plt.pie(
            logic,
            labels=[f"Above {str(border)}", f"Between 0 and {str(border)}",
                    f"Between -{str(border)} and 0", f"Below -{str(border)}"],
            explode=[0.2, 0.1, 0.1, 0.2],
            shadow=True,
            autopct=lambda x: f"{x:.2f}%"
        )
        plt.legend(bbox_to_anchor=(-0.2, 0.1), loc='lower left',
                   borderaxespad=0)

        if idx == 0:
            plt.title(f"Results among hyper-parameters for all {time_frame}"
                      " Algorithms.")

        elif idx == 1:
            plt.title(f"Results among hyper-parameters for {time_frame}"
                      f" GreedyGQ.")

        elif idx == 2:
            plt.title(f"Results among hyper-parameters for {time_frame}"
                      f" QLearn")

        elif idx == 3:
            plt.title(f"Results among hyper-parameters for {time_frame}"
                      f" SARSA")

        plt.tight_layout()
        plt.show()


def savePythonObject(pathAndFileName: str, pythonObject,
                     savingType: str = "pickle") -> None:
    """Function to save a python's object to a file."""
    if savingType == "pickle":
        f = open(pathAndFileName, "wb")
        pickle.dump(pythonObject, f)

    elif savingType == "json":
        savedJson = json.dumps(pythonObject)
        f = open(f"{pathAndFileName}.json", "w")
        f.write(savedJson)

    else:
        raise ValueError(f"Error: savingType = {savingType} not recognized!")


def readPythonObjectFromFile(path: str, openingType: str = "json"):
    """Function to read a file and convert to python's objects."""
    if openingType == "pickle":
        with open(path, 'rb') as f:
            saved = pickle.load(f)
        return saved

    elif openingType == "json":
        with open(path, 'r') as f:
            saved = f.read()
        return json.loads(saved)

    else:
        raise ValueError(f"Error: openingType = {openingType} not recognized!")


def plottingCandlestickChart(OHLCV: pd.DataFrame) -> None:
    """Plotting candlestick chart for OHLCV DataFrame."""
    fig = go.Figure(
        data=[go.Candlestick(
            x=OHLCV.index,
            open=OHLCV['open'],
            high=OHLCV['high'],
            low=OHLCV['low'],
            close=OHLCV['close'])
        ]
    )
    fig.show()


def getOptimal(objects, gains, optimalID=-1):
    """Module to get the optimal model."""
    # ########## pick the best combination of hyper-parameters
    b = np.argsort(gains)
    c = np.sort(gains)

    # ########## get the optimal hyper-parameters and its results
    argBest = int(b[optimalID])

    optimal = {
        "params": objects[0]["params"][argBest],
        "arg": int(argBest),
        "histRprime": np.array(objects[0]["histRprime"][argBest]),
        "meanPL": c[optimalID]
    }

    # ########## merge the histRprime trajectories
    for obt in objects[1:]:
        lastCol = optimal["histRprime"][:, -1][:, None]
        arr1 = np.array(obt["histRprime"][optimal["arg"]]) - 28000 + lastCol
        arr1 = np.hstack([optimal["histRprime"], arr1])
        optimal["histRprime"] = arr1

    return optimal


def run500times(params, files):
    """Module to run 500 times each reinforcement algorithm with different
    seeds."""

    objects = []

    for file in files:

        saved = {
            "params": params,
            "TDErrors": [],
            "histTradePLs": [],
            "sumTradePLs": [],
            "histRprime": [],
            "meanSumTradePLs": []
        }

        for seed in tqdm(range(1, 501)):
            env = Environment(
                n=params[1],
                fileName=file,
                seed=seed,
            )

            if params[0] == "QLearn":
                agent = QLearn(
                    env=env,
                    n=params[1],
                    initInvest=5600 * 5,
                    eta=params[4],
                    gamma=params[5],
                    initType="uniform01",
                    rewardType=params[3],
                    basisFctType=params[2],
                    typeFeatureVector="block",
                    lrScheduler=params[6],
                    verbose=False,
                    seed=seed
                )

            elif params[0] == "GreedyGQ":
                agent = GreedyGQ(
                    env=env,
                    n=params[1],
                    initInvest=5600 * 5,
                    eta=params[4],
                    gamma=params[5],
                    initType="uniform01",
                    rewardType=params[3],
                    zeta=params[6],
                    basisFctType=params[2],
                    typeFeatureVector="block",
                    lrScheduler=params[7],
                    verbose=False,
                    seed=seed
                )

            elif params[0] == "SARSA":
                agent = SARSA(
                    env=env,
                    n=params[1],
                    initInvest=5600 * 5,
                    eta=params[4],
                    gamma=params[5],
                    epsilon=params[6],
                    initType=params[7],
                    rewardType=params[3],
                    basisFctType=params[2],
                    typeFeatureVector="block",
                    lrScheduler=params[8],
                    verbose=False,
                    seed=seed
                )

            while env.terminal is not True:
                agent.run()

            saved["TDErrors"].append(agent.TDErrors)
            saved["histTradePLs"].append(env.histTradePLs)
            saved["sumTradePLs"].append(sum(env.histTradePLs))
            saved["histRprime"].append(env.histRprime)

        saved["meanSumTradePLs"].append(np.mean(saved["sumTradePLs"]))
        objects.append(saved)

    return objects


def optimal500(objects):
    """Module to get the most optimal model among 500 models."""

    optimal = {
        "params": objects[0]["params"],
        "histRprime": np.array(objects[0]["histRprime"]),
        "meanPL": objects[0]["meanSumTradePLs"][0]
    }

    # ########## merge the histRprime trajectories
    for obt in objects[1:]:
        lastCol = optimal["histRprime"][:, -1][:, None]
        arr1 = np.array(obt["histRprime"]) - 28000 + lastCol
        arr1 = np.hstack([optimal["histRprime"], arr1])
        optimal["histRprime"] = arr1
        optimal["meanPL"] += obt["meanSumTradePLs"][0]

    return optimal


def plotReturnTrajectories(
        optimal: dict, initInvest: int = 28000,
        algoType: str = "GreedyGQ", showPlot: bool = True,
        timeFramed: str = "60--min") -> None:
    """Line plot for return trajectories."""

    plt.plot(optimal["histRprime"].T)
    plt.axhline(
        y=initInvest,
        xmin=0,
        xmax=optimal["histRprime"].shape[1],
        color='black',
        linestyle='dotted',
        linewidth=5,
        label=f"Initial investment of {initInvest} points"
    )
    plt.title(f"Return ($G_t$) trajectories for {algoType} "
              f"in {timeFramed} intervals.")
    plt.xlabel("Trading time-steps")
    plt.ylabel("Investment balance in points")
    plt.legend()
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show() if showPlot else None


def plotMeanReturnTrajectory(
        optimal: dict, initInvest: int = 28000, algoType: str = "GreedyGQ",
        showPlot: bool = True, timeFramed: str = "60 min") -> None:
    """Line plot for mean return trajectory."""

    m = optimal["histRprime"].mean(axis=0)
    plt.plot(m)
    plt.axhline(
        y=initInvest,
        xmin=0,
        xmax=m.shape[0],
        color='black',
        linestyle='dotted',
        linewidth=5,
        label=f"Initial investment of {initInvest} points"
    )
    plt.title(f"Mean return trajectory for {algoType} "
              f"in {timeFramed} intervals.")
    plt.xlabel("Trading time-steps")
    plt.ylabel("Investment balance in points")
    plt.legend()
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show() if showPlot else None


def plotDist(
        optimal: dict, initInvest: int = 28000, algoType: str = "GreedyGQ",
        showPlot: bool = True, timeFramed: str = "60 min") -> None:
    """Hist plot for final return trajectories' values."""

    plt.hist(optimal["histRprime"][:, -1], density=True)
    plt.axvline(
        x=initInvest,
        ymin=0,
        ymax=1,
        color='black',
        linestyle='dotted',
        linewidth=5,
        label=f"Initial capital of {initInvest} points"
    )
    plt.title(f"Return distribution for {algoType} "
              f"in {timeFramed} intervals.")
    plt.xlabel("Return in points")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show() if showPlot else None


def plotBox(
        optimal: dict, initInvest: int = 28000, algoType: str = "GreedyGQ",
        showPlot: bool = True, timeFramed: str = "60 min") -> None:
    """Box and Swarm plot for final return trajectories."""

    sns.boxplot(
        optimal["histRprime"][:, -1],
        linewidth=2.5
    )
    sns.swarmplot(
        data=optimal["histRprime"][:, -1],
        color=".25",
        orient="h"
    )

    plt.axvline(
        x=initInvest,
        ymin=0,
        ymax=1,
        color='black',
        linestyle='dotted',
        linewidth=5,
        label=f"Initial capital of {initInvest} points"
    )

    plt.title(f"Final returns for {algoType} "
              f"in {timeFramed} intervals.")
    plt.xlabel("Return in points")
    plt.ylabel(f"Frequency")
    plt.legend()
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show() if showPlot else None


def plotAllBoxes(
        data, initInvest: int = 28000, showPlot: bool = True) -> None:
    """Box plot for all final return trajectories of all algorithms."""

    sns.boxplot(
        data=data,
        linewidth=2.5,
        orient='h'
    )

    plt.axvline(
        x=initInvest,
        ymin=0,
        ymax=1,
        color='black',
        linestyle='dotted',
        linewidth=5,
        label=f"28k points"
    )

    plt.title(f"Final returns for all seeds of all optimal algorithms.")
    plt.xlabel("Return in points")
    plt.legend()
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show() if showPlot else None
