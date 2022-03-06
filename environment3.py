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
        self.histTradePL = []
        self.lastHistTradePL = []
        self.countNotExecuted = 0

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

        self.terminal = True if len(primeS) == self.lenData else False

        C = S.iloc[-1, 3]
        primeC = primeS.iloc[-1, 3]

        if wasRandEpsilon is False:
            if A == -1:                             # limit short
                if self.tradeStatus == 0:           # new trade opened
                    self.tau += 1
                    self.tradeStatus = -1
                    primeR = C - primeC
                    self.entryPrice = 1 * primeS.iloc[-1 - self.tau, 3]
                    tradePL = self.entryPrice - primeC
                    self.histR.append(primeR)
                    self.histTradePL.append(tradePL)
                    lnTradePL = self.ln(primeC, abs(self.entryPrice))
                    return (primeS, primeR, self.entryPrice, tradePL,
                            self.tau, self.tradeStatus, lnTradePL)

                elif self.tradeStatus == 1:     # current trade closed
                    self.tradeStatus = 0
                    primeR = primeC - C
                    self.entryPrice = 0
                    tradePL = 0
                    self.tau = 0
                    self.histR.append(primeR)
                    self.lastHistTradePL = self.histTradePL
                    self.histTradePL = []
                    lnTradePL = 0
                    return (primeS, primeR, self.entryPrice, tradePL,
                            self.tau, self.tradeStatus, lnTradePL)

            elif A == 0:                            # do nothing
                if self.tradeStatus == 0:           # not on a trade
                    primeR, tradePL, lnTradePL = 0.0, 0.0, 0.0
                    self.histR.append(primeR)
                    return (primeS, primeR, self.entryPrice, tradePL,
                            self.tau, self.tradeStatus, lnTradePL)

                elif self.tradeStatus == 1:         # on a short trade
                    self.tau += 1
                    primeR = primeC - C
                    tradePL = self.entryPrice + primeC
                    self.histR.append(primeR)
                    self.histTradePL.append(tradePL)
                    lnTradePL = self.ln(primeC, abs(self.entryPrice))
                    return (primeS, primeR, self.entryPrice, tradePL,
                            self.tau, self.tradeStatus, lnTradePL)

                elif self.tradeStatus == -1:        # on a long trade
                    self.tau += 1
                    primeR = C - primeC
                    tradePL = self.entryPrice - primeC
                    self.histR.append(primeR)
                    self.histTradePL.append(tradePL)
                    lnTradePL = self.ln(primeC, abs(self.entryPrice))
                    return (primeS, primeR, self.entryPrice, tradePL,
                            self.tau, self.tradeStatus, lnTradePL)

            elif A == 1:                            # limit long
                if self.tradeStatus == 0:           # new trade opened
                    self.tau += 1
                    self.tradeStatus = 1
                    primeR = primeC - C
                    self.entryPrice = -1 * primeS.iloc[-1 - self.tau, 3]
                    tradePL = self.entryPrice + primeC
                    tau = self.tau
                    self.histR.append(primeR)
                    self.histTradePL.append(tradePL)
                    lnTradePL = self.ln(primeC, abs(self.entryPrice))
                    return (primeS, primeR, self.entryPrice, tradePL,
                            tau, self.tradeStatus, lnTradePL)

                elif self.tradeStatus == -1:    # current short trade closed
                    self.tradeStatus = 0
                    primeR = C - primeC
                    self.entryPrice = 0
                    tradePL = 0
                    self.tau = 0
                    self.histR.append(primeR)
                    self.lastHistTradePL = self.histTradePL
                    self.histTradePL = []
                    lnTradePL = 0
                    return (primeS, primeR, self.entryPrice, tradePL,
                            self.tau, self.tradeStatus, lnTradePL)

        elif wasRandEpsilon is True:
            tradePL, lnTradePL = 0.0, 0.0

            if A == -1:                         # limit short
                primeR = C - primeC
                self.histR.append(primeR)
                if (self.tradeStatus == -1) or (self.tradeStatus == 1):
                    self.tau += 1

                return (primeS, primeR, self.entryPrice, tradePL,
                        self.tau, self.tradeStatus, lnTradePL)

            elif A == 0:                        # do nothing
                primeR = 0.0
                self.histR.append(primeR)
                if (self.tradeStatus == -1) or (self.tradeStatus == 1):
                    self.tau += 1

                return (primeS, primeR, self.entryPrice, tradePL,
                        self.tau, self.tradeStatus, lnTradePL)

            elif A == 1:                        # limit long
                primeR = primeC - C
                self.histR.append(primeR)
                if (self.tradeStatus == -1) or (self.tradeStatus == 1):
                    self.tau += 1

                return (primeS, primeR, self.entryPrice, tradePL,
                        self.tau, self.tradeStatus, lnTradePL)
