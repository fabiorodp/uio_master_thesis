# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
import torch as tr
import pandas as pd
from readData import get_instrument, create_candles

import warnings

# suppress warnings
warnings.filterwarnings('ignore')


class Environment:

    def __init__(self, n, fileName="data/WINZ21_1min_OLHCV.csv"):
        self.data = pd.read_csv(f"{fileName}", sep=";")
        self.t = 1
        self.n = n
        self.minusA = 0
        self.tau = 0

    def getCurrentState(self):
        return self.data.iloc[0: self.n + self.t, :]

    def getNextState(self, A):
        self.t += 1
        S = self.data.iloc[0: self.n + self.t - 1, :]
        primeS = self.data.iloc[0: self.n + self.t, :]

        C = S.iloc[-1, 3]
        primeH = primeS.iloc[-1, 1]
        primeL = primeS.iloc[-1, 2]
        primeC = primeS.iloc[-1, 3]

        if A == -1:                     # limit short

            if primeH > C:              # short trade executed
                self.tau += 1

                if self.minusA == 0:    # new trade opened
                    self.minusA = -1
                    primeR = C - primeC
                    entryPrice = primeS.iloc[-1 - (self.tau - 1), 3]
                    lineR = entryPrice - primeC
                    tau = self.tau
                    return primeS, primeR, entryPrice, lineR, tau, self.minusA

                elif self.minusA == 1:  # current trade closed
                    self.minusA = 0
                    primeR = primeC - C
                    entryPrice = primeS.iloc[-1 - (self.tau - 1), 3]
                    lineR = primeC - entryPrice
                    tau = self.tau
                    self.tau = 0
                    return primeS, primeR, entryPrice, lineR, tau, self.minusA

            else:                       # trade not executed
                primeR = 0.0
                entryPrice = 0.0
                lineR = 0.0
                tau = self.tau
                return primeS, primeR, entryPrice, lineR, tau, self.minusA

        elif A == 0:                    # do nothing

            if self.minusA == 0:        # not on a trade
                primeR = 0.0
                entryPrice = 0.0
                lineR = 0.0
                tau = self.tau
                return primeS, primeR, entryPrice, lineR, tau, self.minusA

            elif self.minusA == -1:     # on a short trade
                self.tau += 1
                primeR = C - primeC
                entryPrice = primeS.iloc[-1 - (self.tau - 1), 3]
                lineR = entryPrice - primeC
                tau = self.tau
                return primeS, primeR, entryPrice, lineR, tau, self.minusA

            elif self.minusA == 1:      # on a long trade
                self.tau += 1
                primeR = primeC - C
                entryPrice = primeS.iloc[-1 - (self.tau - 1), 3]
                lineR = primeC - entryPrice
                tau = self.tau
                return primeS, primeR, entryPrice, lineR, tau, self.minusA

        elif A == 1:                    # limit long

            if primeL < C:              # long trade executed
                self.tau += 1

                if self.minusA == 0:    # new trade opened
                    self.minusA = 1
                    primeR = primeC - C
                    entryPrice = primeS.iloc[-1 - (self.tau - 1), 3]
                    lineR = primeC - entryPrice
                    tau = self.tau
                    return primeS, primeR, entryPrice, lineR, tau, self.minusA

                elif self.minusA == -1:  # current trade closed
                    self.minusA = 0
                    primeR = C - primeC
                    entryPrice = primeS.iloc[-1 - (self.tau - 1), 3]
                    lineR = entryPrice - primeC
                    tau = self.tau
                    self.tau = 0
                    return primeS, primeR, entryPrice, lineR, tau, self.minusA

            else:                       # trade not executed
                primeR = 0.0
                entryPrice = 0.0
                lineR = 0.0
                tau = self.tau
                return primeS, primeR, entryPrice, lineR, tau, self.minusA

        else:
            raise ValueError(f"ERROR: A = {A} not recognized.")


class Agent:

    @staticmethod
    def l(previousPrice: tr.Tensor,
          currentPrice: tr.Tensor) -> float:
        """Computing a feature..."""
        return tr.log(currentPrice / previousPrice).item()

    @staticmethod
    def basisFunction(x: float, a: float = 2, b: float = 1,
                      c: float = 10 ** 15, d: float = -1) -> float:
        """Basis function."""
        return (a / (1 + b * np.exp(-c * x))) - d

    @staticmethod
    def rewardFunction(Gtplus1, rewardType):
        """Reward function."""
        if rewardType == "shapeRatio":
            return np.mean(Gtplus1) / np.sqrt(np.var(Gtplus1))
        else:
            raise ValueError(f"ERROR: rewardType {rewardType} "
                             f"not recognized...")

    def __init__(self, env, n, eta=0.05, gamma=0.95, epsilon=0.1,
                 initType="zeros", rewardType="immediate",
                 algoType="QL", seed=0):

        # agent's parameters
        self.env = env
        self.n = n
        self.eta = eta                          # learning rate
        self.gamma = gamma                      # discount factor
        self.epsilon = epsilon                  # epsilon for e-greedy policy
        self.initType = initType                # init for w, b, f
        self.rewardType = rewardType            # for reward function
        self.algoType = algoType                # sarsa, ql, or g-ql

        # seeding the experiment
        if seed != 0:
            self.seed = seed
            np.random.seed(self.seed)
            tr.manual_seed(self.seed)

        # variables
        self.spaceA = [-1, 0, 1]
        self.S = 0
        self.minusA, self.A = 0, 0
        self.R, self.primeR, self.lineR = 0, 0, 0
        self.Q, self.nablaQ = 0, 0
        self.zeroVector = tr.zeros((1, n + 1), dtype=tr.double)
        self.randEpsilon = 0
        self.delta = 0                              # TD-error
        self.t, self.tau = 1, 0
        self.d = len(self.spaceA) * (self.n + 1)

        # init vectors
        if initType == "zeros":
            self.w = tr.zeros((self.d, 1), dtype=tr.double)
        else:
            raise ValueError("ERROR: initType not recognized!")

        # self.memoryBase = pd.DataFrame(columns=[1, 2])
        # col = ["open", "high", "low", "close", "volume", "minusA",
        #        "A", "lineR", "primeR", "Q", "nablaQ", "primeO",
        #        "primeH", "primeL", "primeC", "primeV"]
        # colIdx = [i for i in range(len(col))]
        """self.memory = {
            "open": {}, "high": {}, "low": {}, "close": {}, "volume": {},
            "minusA": {}, "A": {}, "lineR": {}, "primeR": {}, "Q": {},
            "nablaQ": {}
        }"""
        self.memory = pd.DataFrame(
            columns=['open', 'high', 'low', 'close', 'volume',
                     'minusA', 'A', 'tau', 'lineR', 'R', 'primeR',
                     'optQ', 'TD-error']
        )
        self.memoryW = None
        self.memoryNablaQ = None

    def getBasisVector(self, S, minusA, lineR):
        b = tr.zeros((1, self.n + 1), dtype=tr.double)

        for i in range(1, self.n + 1):
            currentPrice = S[self.t - (self.n - 1) + i]
            previousPrice = S[self.t - self.n + i]

            b[0, i - 1] = self.basisFunction(
                x=self.l(
                    previousPrice=previousPrice,
                    currentPrice=currentPrice
                )
            )

        if minusA == 0:
            b[0, -1] = self.basisFunction(x=0)

        else:
            b[0, -1] = self.basisFunction(x=lineR)

        return b

    def getFeatureVector(self, S, A, minusA, lineR):

        b = self.getBasisVector(
            S=S,
            minusA=minusA,
            lineR=lineR
        )

        f = tr.zeros((self.d, 1), dtype=tr.double)

        if A == -1:
            f = tr.hstack(
                (b, self.zeroVector, self.zeroVector)
            ).T
        elif A == 0:
            f = tr.hstack(
                (self.zeroVector, b, self.zeroVector)
            ).T
        elif A == 1:
            f = tr.hstack(
                (self.zeroVector, self.zeroVector, b)
            ).T

        return f

    def epsilonGreedyPolicy(self, S, As):
        self.randEpsilon = np.random.uniform(low=0, high=1, size=None)
        if self.epsilon <= self.randEpsilon:
            a = np.random.choice(
                As,
                size=None,
                replace=False,
                p=None
            )
            f = self.getFeatureVector(
                S=S,                        # current state
                A=a,                        # action t
                minusA=self.minusA,         # action t-1
                lineR=self.lineR            # current trade profit
            )
            q = (self.w.T @ f).item()
            return a, q, f

        else:
            Q, nablaQ = {}, {}
            for a in As:
                f = self.getFeatureVector(
                    S=S,                    # current state
                    A=a,                    # action t
                    minusA=self.minusA,     # action t-1
                    lineR=self.lineR        # current trade profit
                )
                Q[a] = (self.w.T @ f).item()
                nablaQ[a] = f

            argmax = max(Q, key=Q.get)
            # TODO: maybe it is choosing the wrong one when even.

            maxQ = max(Q.values())
            return argmax, maxQ, nablaQ[argmax]

    def spaceAs(self):
        if self.minusA == -1:
            return [0, 1]

        elif self.minusA == 0:
            return [-1, 0, 1]

        elif self.minusA == 1:
            return [-1, 0]

    def reward(self, S, primeS):
        if self.rewardType == "immediate":
            return primeS[-1].item() - S[-1].item()

        elif self.rewardType == "logReturn":
            return self.l(previousPrice=S[-1], currentPrice=primeS[-1])

        else:
            raise ValueError(f"ERROR: rewardType {self.rewardType} "
                             f"not recognized.")

    def saveMemory(self, dfS, minusA, A, tau, lineR, R, primeR, Q,
                   delta, nablaQ):

        col = dfS.columns.to_list()
        extCols = ["minusA", "A", "tau", "lineR", "R", "primeR",
                   "optQ", "TD-error"]
        keys = col+extCols

        val1 = [dfS[k][-1] for k in dfS.keys().to_list()]
        val2 = [minusA, A, tau, lineR, R, primeR, Q, delta]
        vals = val1+val2

        timeIdx= dfS.index.to_list()[-1]

        memory = pd.DataFrame(vals).T
        memory.columns = keys
        memory.index = [timeIdx]
        self.memory = pd.concat([self.memory, memory], axis=0)

        if self.memoryW is None:
            mem = self.w.T.tolist()
            df = pd.DataFrame(mem)
            df.index = [self.memory.index.to_list()[-1]]
            self.memoryW = df

        else:
            mem = self.w.T.tolist()
            df = pd.DataFrame(mem)
            df.index = [self.memory.index.to_list()[-1]]
            self.memoryW = pd.concat([self.memoryW, df], axis=0)

        if self.memoryNablaQ is None:
            mem = nablaQ.T.tolist()
            df = pd.DataFrame(mem)
            df.index = [self.memory.index.to_list()[-1]]
            self.memoryNablaQ = df

        else:
            mem = nablaQ.T.tolist()
            df = pd.DataFrame(mem)
            df.index = [self.memory.index.to_list()[-1]]
            self.memoryNablaQ = pd.concat([self.memoryNablaQ, df], axis=0)

    def run(self):

        if self.t == 1:
            dfS, A = self.env.getCurrentState(), self.A
            S = tr.from_numpy(dfS.values[:, 3])[:, None]

            f = self.getFeatureVector(
                S=S,                        # current state
                A=A,                        # action t
                minusA=self.minusA,         # action t-1
                lineR=self.lineR            # current trade profit
            )

            Q = (self.w.T @ f).item()
            nablaQ = f

            self.saveMemory(
                dfS=dfS,
                minusA=self.minusA,
                A=A,
                tau=self.tau,
                lineR=self.lineR,
                R=self.R,
                primeR=self.primeR,
                Q=Q,
                delta=self.delta,
                nablaQ=nablaQ
            )

        else:
            S, A, Q, nablaQ = self.S, self.A, self.Q, self.nablaQ

        dfPrimeS, primeR, entryPrice, self.lineR, self.tau, self.minusA = \
            self.env.getNextState(A)

        primeS = tr.from_numpy(dfPrimeS.values[:, 3])[:, None]

        primeA, primeQ, primeNablaQ = self.epsilonGreedyPolicy(
            S=primeS,
            As=self.spaceAs()
        )

        delta = primeR + primeQ - Q                 # TD-error
        print(f"Loss {delta}")

        # TODO: primeR muito alto por causa de ser imediato.
        #  Tentar shape ou log-return.

        self.w += self.eta * delta * nablaQ         # weight

        self.saveMemory(
            dfS=dfPrimeS,
            minusA=self.minusA,
            A=primeA,
            tau=self.tau,
            lineR=self.lineR,
            R=self.R,
            primeR=primeR,
            Q=primeQ,
            delta=delta,
            nablaQ=primeNablaQ
        )

        self.t += 1
        self.S = primeS
        self.A = primeA
        self.Q, self.nablaQ = primeQ, primeNablaQ
        self.delta = delta
        self.R = primeR


if __name__ == '__main__':
    agent = Agent(
        env=Environment(n=2),
        n=2,
        eta=0.001,
        gamma=0.95,
        epsilon=0.1,
        initType="zeros",
        rewardType="immediate",
        algoType="SARSA",
        seed=1
    )

    agent.run()
