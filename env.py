# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
import pandas as pd
from datetime import datetime
from technicalAnalysis import bollingerBands, ema, lwma

# suppress warnings
import warnings

warnings.filterwarnings('ignore')


class Environment:

    @staticmethod
    def applyTA(data, freq, std_value=2.0, column_base='close'):
        data = ema(
            data=data,
            freq=freq,
            column_base=column_base
        )

        data = bollingerBands(
            data=data,
            freq=freq,
            std_value=std_value,
            column_base=column_base
        )

        data = lwma(
            data=data,
            freq=freq,
            column_base=column_base
        )
        return data

    @staticmethod
    def ln(currentPrice, previousPrice):
        """Computing a feature..."""
        return np.log(currentPrice / previousPrice)

    @staticmethod
    def cleanCurrentTrade():
        return {
            "time": [],
            "entryPrice": 0,
            "currentTradePLs": [0],
            "histTradePLs": [],
            "lnTradePLs": [],
        }

    def __init__(self, n, fileName="data/WING22_1min_OLHCV.csv",
                 initInvest=0, seed=0):
        # seeding the experiment
        self.seed = seed
        if seed != 0:
            self.seed = seed
            np.random.seed(self.seed)

        self.n = n
        self.fileName = fileName
        self.initInvest = initInvest
        self.t = 1
        self.data = pd.read_csv(f"{self.fileName}", sep=";")

        self.S = self.data.iloc[: self.n + self.t, :]

        self.S = self.applyTA(
            data=self.S,
            freq=self.n + 1,
            std_value=2.0,
            column_base='close'
        )

        self.terminal = True if len(self.S) == len(self.data) else False

        self.entryPrice = 0
        self.tradeRandEpsilon = False
        self.tradeStatus = 0
        self.Rprime = self.initInvest
        self.tradePL = 0
        self.lnTradePL = 0

        self.tradeMemory = {
            "time": [],
            "entryPrice": 0,
            "currentTradePLs": [0],
            "histTradePLs": [],
            "lnTradePLs": [],
        }

        self.histTradeMemory = []
        self.histRprime = []
        self.histTradePLs = []

    def runNext(self, A):
        Sprime = self.data.iloc[: self.n + 1 + self.t, :]  # n=2 + 1 + t=1
        Sprime = self.applyTA(
            data=Sprime,
            freq=self.n + 1,
            std_value=2.0,
            column_base='close'
        )

        C = Sprime.iloc[-2, 3]
        Cprime = Sprime.iloc[-1, 3]

        try:
            timePrime = datetime.strptime(Sprime.index[-1],
                                          '%Y-%m-%d %H:%M:%S')
        except:
            timePrime = datetime.strptime(Sprime.index[-1],
                                          '%Y-%m-%d %H:%M:%S.%f')
        print(timePrime)

        if A == -1:  # limit short
            if self.tradeStatus == 0:  # new trade opened
                self.tradeStatus = -1
                self.entryPrice = C
                tradePL = self.entryPrice - Cprime
                deltaTradePLs = tradePL - self.tradePL
                self.Rprime += deltaTradePLs
                self.tradePL = tradePL
                self.lnTradePL = self.ln(Cprime, abs(self.entryPrice))

                self.saveNewTrade(
                    timePrime=timePrime,
                    entryPrice=self.entryPrice,
                    tradePL=self.tradePL,
                    lnTradePL=self.lnTradePL
                )

            elif self.tradeStatus == 1:  # current trade closed
                self.histTradePLs.append(self.tradePL)
                self.tradeStatus = 0
                self.entryPrice = 0
                self.tradePL = 0
                self.Rprime += self.tradePL
                self.lnTradePL = 0
                self.histTradeMemory.append(self.tradeMemory)
                self.tradeMemory = self.cleanCurrentTrade()

        elif A == 0:  # do nothing
            if self.tradeStatus == 1:  # on a short trade
                tradePL = self.entryPrice + Cprime
                deltaTradePLs = tradePL - self.tradePL
                self.Rprime += deltaTradePLs
                self.tradePL = tradePL
                self.lnTradePL = self.ln(Cprime, abs(self.entryPrice))

                self.saveUpdateTrade(
                    timePrime=timePrime,
                    tradePL=self.tradePL,
                    lnTradePL=self.lnTradePL
                )

            elif self.tradeStatus == -1:  # on a long trade
                tradePL = self.entryPrice - Cprime
                deltaTradePLS = tradePL - self.tradePL
                self.Rprime += deltaTradePLS
                self.tradePL = tradePL
                self.lnTradePL = self.ln(Cprime, abs(self.entryPrice))

                self.saveUpdateTrade(
                    timePrime=timePrime,
                    tradePL=self.tradePL,
                    lnTradePL=self.lnTradePL
                )

        elif A == 1:  # limit long
            if self.tradeStatus == 0:  # new trade opened
                self.tradeStatus = 1
                self.entryPrice = -C
                tradePL = self.entryPrice + Cprime
                deltaTradePLs = tradePL - self.tradePL
                self.Rprime += deltaTradePLs
                self.tradePL = tradePL
                self.lnTradePL = self.ln(Cprime, abs(self.entryPrice))

                self.saveNewTrade(
                    timePrime=timePrime,
                    entryPrice=self.entryPrice,
                    tradePL=self.tradePL,
                    lnTradePL=self.lnTradePL
                )

            elif self.tradeStatus == -1:  # current trade closed
                self.histTradePLs.append(self.tradePL)
                self.tradeStatus = 0
                self.entryPrice = 0
                self.tradePL = 0
                self.Rprime += self.tradePL
                self.lnTradePL = 0
                self.histTradeMemory.append(self.tradeMemory)
                self.tradeMemory = self.cleanCurrentTrade()

        if len(Sprime) == len(self.data):
            self.terminal = True
            if self.tradeStatus != 0:
                self.histTradePLs.append(self.tradePL)

        self.histRprime.append(self.Rprime)
        self.S = Sprime
        self.t += 1

    def saveNewTrade(self, timePrime, entryPrice, tradePL, lnTradePL):
        self.tradeMemory["time"].append(timePrime)
        self.tradeMemory["entryPrice"] = entryPrice
        self.tradeMemory["currentTradePLs"].append(tradePL)
        self.tradeMemory["lnTradePLs"].append(lnTradePL)

    def saveUpdateTrade(self, timePrime, tradePL, lnTradePL):
        self.tradeMemory["time"].append(timePrime)
        self.tradeMemory["currentTradePLs"].append(tradePL)
        self.tradeMemory["lnTradePLs"].append(lnTradePL)