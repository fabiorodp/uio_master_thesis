# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
import pandas as pd
import warnings

# suppress warnings
# warnings.filterwarnings('ignore')


class Environment:

    @staticmethod
    def ln(currentPrice, previousPrice):
        """Computing a feature..."""
        return np.log(currentPrice / previousPrice)

    def __init__(self, n, fileName="data/WING22_1min_OLHCV.csv"):
        self.data = pd.read_csv(f"{fileName}", sep=";")
        self.t = 1
        self.n = n
        self.tradeStatus = 0
        self.tau = 0
        self.terminal = False
        self.lenData = len(self.data)
        self.entryPrice = 0
        self.histR = []
        self.portfolioPLs = []
        self.tradePL = 0

    def getCurrentState(self):
        S = self.data.iloc[0: self.n + self.t, :]
        self.terminal = True if len(S) == self.lenData else False
        return S

    def getNextState(self, A, wasRandEpsilon=False):
        self.t += 1
        S = self.data.iloc[0: self.n + self.t - 1, :]
        primeS = self.data.iloc[0: self.n + self.t, :]

        self.S = S
        self.primeS = primeS

        C = S.iloc[-1, 3]
        primeC = primeS.iloc[-1, 3]

        if wasRandEpsilon is False:
            # here A and tradeStatus are dependent variables
            if A == -1:                             # limit short
                if self.tradeStatus == 0:           # new trade opened
                    self.tau += 1
                    self.tradeStatus = -1
                    primeR = C - primeC
                    self.entryPrice = 1 * primeS.iloc[-1 - self.tau, 3]
                    self.tradePL = self.entryPrice - primeC
                    lnTradePL = self.ln(primeC, abs(self.entryPrice))

                elif self.tradeStatus == 1:         # current trade closed
                    primeR = primeC - C
                    self.portfolioPLs.append(self.tradePL)
                    self.tradeStatus = 0
                    self.entryPrice = 0
                    self.tradePL = 0
                    self.tau = 0
                    lnTradePL = 0

            elif A == 0:                            # do nothing
                if self.tradeStatus == 0:           # not on a trade
                    primeR, self.tradePL, lnTradePL = 0.0, 0.0, 0.0

                elif self.tradeStatus == 1:         # on a short trade
                    self.tau += 1
                    primeR = primeC - C
                    self.tradePL = self.entryPrice + primeC
                    lnTradePL = self.ln(primeC, abs(self.entryPrice))

                elif self.tradeStatus == -1:        # on a long trade
                    self.tau += 1
                    primeR = C - primeC
                    self.tradePL = self.entryPrice - primeC
                    lnTradePL = self.ln(primeC, abs(self.entryPrice))

            elif A == 1:                            # limit long
                if self.tradeStatus == 0:           # new trade opened
                    self.tau += 1
                    self.tradeStatus = 1
                    primeR = primeC - C
                    self.entryPrice = -1 * primeS.iloc[-1 - self.tau, 3]
                    self.tradePL = self.entryPrice + primeC
                    lnTradePL = self.ln(primeC, abs(self.entryPrice))

                elif self.tradeStatus == -1:    # current short trade closed
                    primeR = C - primeC
                    self.portfolioPLs.append(self.tradePL)
                    self.tradeStatus = 0
                    self.entryPrice = 0
                    self.tradePL = 0
                    self.tau = 0
                    lnTradePL = 0

            self.histR.append(primeR)

            if len(primeS) == self.lenData:
                self.terminal = True
                if self.tradeStatus != 0:
                    self.portfolioPLs.append(self.tradePL)

            return (primeS, primeR, self.entryPrice, self.tradePL,
                    self.tau, self.tradeStatus, lnTradePL)

        elif wasRandEpsilon is True:

            if self.tradeStatus == -1:          # on a short trade
                self.tradePL = self.entryPrice - primeC
                lnTradePL = self.ln(primeC, abs(self.entryPrice))
                self.tau += 1

            elif self.tradeStatus == 0:         # out of the market
                self.tradePL, lnTradePL = 0.0, 0.0

            elif self.tradeStatus == 1:         # on a long trade
                self.tradePL = self.entryPrice + primeC
                lnTradePL = self.ln(primeC, abs(self.entryPrice))
                self.tau += 1

            # here A and tradeStatus are independent variables
            if A == -1:                         # limit short
                primeR = C - primeC

            elif A == 0:                        # do nothing
                primeR = 0.0

            elif A == 1:                        # limit long
                primeR = primeC - C

            self.histR.append(primeR)

            if len(primeS) == self.lenData:
                self.terminal = True
                if self.tradeStatus != 0:
                    self.portfolioPLs.append(self.tradePL)

            return (primeS, primeR, self.entryPrice, self.tradePL,
                    self.tau, self.tradeStatus, lnTradePL)
