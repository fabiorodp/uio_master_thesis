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

# setting parent directory to be accessed
# os.chdir('..')


def savePythonObject(pathAndFileName: str, pythonObject,
                     savingType: str = "pickle") -> None:
    """Function to save a python's object to a file..."""
    if savingType == "pickle":
        f = open(pathAndFileName, "wb")
        pickle.dump(pythonObject, f)

    elif savingType == "json":
        savedJson = json.dumps(pythonObject)
        f = open(f"{pathAndFileName}.json", "w")
        f.write(savedJson)

    else:
        raise ValueError(f"Error: savingType = {savingType} not recognized!")


def readPythonObjectFromFile(path: str, openingType: str = "pickle"):
    """Function to read a file and convert to python's objects..."""
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
    """Plotting candlestick chart for OHLCV DataFrame..."""
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


def getAnAsset(ticker='PETR4', in_folder='data/', out_folder='data/'):
    """
    This is a function to extract all negotiations of a ticker.
    Parameters:
    ===================
    :param ticker: str: The financial security symbol.
    :param in_folder: str: The Path where the data files are stored.
    :param out_folder: str: The Path where the CSV data file will be stored.
    Return:
    ===================
    It does not return anything, only saves the filtered data by the
    given security ticker.
    """
    for filename in os.listdir(in_folder):
        print(filename)
        if filename.endswith(".zip") and \
                filename.startswith('TradeIntraday_'):
            # importing data-set file
            data = pd.read_csv(in_folder + filename, compression='zip',
                               sep=';', header=0, dtype=str)

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
            data["DateTime"] = pd.to_datetime(data["DateTime"],
                                              format='%Y-%m-%d %H%M%f')

            # replacing "," to "." in price
            data.columns = ["Price", "Volume", "DateTime"]
            data["Price"] = data["Price"].str.replace(',', '.')

            # fixing dtypes
            data["Price"] = data["Price"].astype(np.float64)
            data["Volume"] = data["Volume"].astype(np.int64)

            # dropping old index
            data.reset_index(inplace=True, drop='index')

            # creating csv data file
            data.to_csv(
                f'{out_folder}/{ticker}_{filename[14:-6]}.csv', sep=';',
                index_label=False)


def parseIntoTimeBars(ticker='PETR4', candles_periodicity='1D',
                      in_folder='data/', out_folder=None):
    """
    This is a function to create candles data based on the ticker
    Parameters:
    ===================
    :param ticker: str: The financial instrument ticker. Default: 'PETR4'.
    :param candles_periodicity: str: Periodicity of the candle. Default
                                     '1D' that means 1 day. Options: 'xmin'
                                     where x is the number of minutes.
    :param in_folder: str: The folder where the data file containing all
                           negotiations of the ticker is stored.
    :param out_folder: str: The Path where the CSV data file will be stored.
    Return:
    ===================
    :returns data: pd.DataFrame: DataFrame containing the OLHCV data for
                                 the given ticker and periodicity.
    """
    data = pd.DataFrame()

    for file in os.listdir(in_folder):
        print(file)

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
    if out_folder is not None:
        data.to_csv(
            f'{out_folder}{ticker}_{candles_periodicity}_OLHCV.csv', sep=';',
            index_label=False)

    return data


def parseIntoTickBars(ticker='WING22', numTicks=15000,
                      in_folder='../data/WING22/CSV/', out_folder=None):

    data = pd.DataFrame()
    for file in os.listdir(in_folder):
        print(file)

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
    if out_folder is not None:
        dfFinal.to_csv(
            f'{out_folder}{ticker}_{numTicks}ticks_OLHCV.csv', sep=';',
            index_label=False)

    return dfFinal


def loadResults(files):

    objects, gains = [], None

    for idx, file in enumerate(files):

        temp = readPythonObjectFromFile(
            path=file,
            openingType="json"
        )

        objects.append(temp)

        if idx == 0:
            gains = np.array(temp["meanSumTradePLs"])
        else:
            gains += np.array(temp["meanSumTradePLs"])

    return objects, gains


def topWorstBestAndOptimal(top, objects, gains):

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

    argTopBest = [a for a in b[-top:]]
    topBest = {
        "args": argTopBest,
        "params": [[a for a in objects[0]["params"][i]]
                   for i in argTopBest],
        "scores": [a for a in c[-top:]]
    }

    return topWorst, topBest


def getOptimal(objects, gains, optimalID=-1):

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


def plotReturnTrajectories(
        optimal: dict, initInvest: int = 28000,
        numSeeds: int = 50, showPlot: bool = True) -> None:
    """Line plot for return trajectories..."""

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
    plt.title(f"Return ($G_t$) trajectories for {numSeeds} different seeds")
    plt.xlabel("Trading time-steps")
    plt.ylabel("Investment balance in points")
    plt.legend()
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show() if showPlot else None


def plotMeanReturnTrajectory(
        optimal: dict, initInvest: int = 28000, numSeeds: int = 50,
        showPlot: bool = True) -> None:
    """Line plot for mean return trajectory..."""

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
    plt.title(f"Mean Return ($G_t$) trajectory for {numSeeds} different seeds")
    plt.xlabel("Trading time-steps")
    plt.ylabel("Investment balance in points")
    plt.legend()
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show() if showPlot else None


if __name__ == '__main__':
    getAnAsset(
        ticker='WINM21',
        in_folder='data/WINZ21/',
        out_folder='data/WINZ21/CSV'
    )

    parseIntoTimeBars(
        ticker='WINZ21',
        candles_periodicity='60min',
        in_folder='data/WINZ21/CSV/',
        out_folder='data/WINZ21/'
    )

    parseIntoTickBars(
        ticker='WING22',
        numTicks=250000,
        in_folder='../data/WING22/CSV/',
        out_folder='../data/WING22/'
    )

    """
    savePythonObject(
        pathAndFileName="results/WINQ21/saved_sigmoid_WINQ21_250000ticks",
        pythonObject=saved,
        savingType="json"
    )
    """
