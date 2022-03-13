# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
import torch as tr
import pandas as pd
from env1 import Environment
from datetime import datetime

import warnings

# suppress warnings
# warnings.filterwarnings('ignore')


class Agent:

    @staticmethod
    def ln(currentPrice: tr.Tensor, previousPrice: tr.Tensor) -> float:
        """Computing a feature..."""
        return tr.log(currentPrice / previousPrice).item()

    @staticmethod
    def basisFunction(x: float, basisFctType: str = "hypTanh123") -> float:
        """Basis function."""
        if basisFctType == "hypTanh123":
            a, b, c, d = 2, 1, 10**15, -1
            return (a / (1 + b * np.exp(-c * x))) - d

        elif basisFctType == "hypTanh":
            return np.tanh(x)

        elif basisFctType == "relu":
            return np.max([0, x])

        elif basisFctType == "sigmoid":
            return 1 / 1 + np.exp(-x)

        else:
            raise ValueError(f"basisFctType = {basisFctType} not recognized!")

    @staticmethod
    def rewardFunction(Gtplus1, rewardType):
        """Reward function."""
        if rewardType == "shapeRatio":
            r = np.mean(Gtplus1) / np.sqrt(np.var(Gtplus1))
            r = 0 if np.isnan(r) else r
            return r

        elif rewardType == "mean":
            return np.mean(Gtplus1)

        elif rewardType == "sum":
            return np.sum(Gtplus1)

        else:
            raise ValueError(f"ERROR: rewardType {rewardType} "
                             f"not recognized...")

    @staticmethod
    def verbose(verbose, dfPrimeS, tradeStatus, primeTradeStatus, A, primeA,
                tau, primeTau, entryPrice, primeTradePL, lnPrimeTradePL,
                primeR):
        if verbose is True:
            print(f"\nThe prime time is {dfPrimeS.index.to_list()[-1]}.")
            print(f"The prime closed price is {dfPrimeS.iloc[-1, 3]}.")
            print(f"\ntradeStatus = {tradeStatus}.")
            print(f"primeTradeStatus = {primeTradeStatus}.")
            print(f"A = {A}.")
            print(f"primeA = {primeA}.")
            print(f"tau = {tau}.")
            print(f"primeTau = {primeTau}.")
            print(f"entryPrice = {entryPrice}.")
            print(f"primeTradePL = {primeTradePL}.")
            print(f"lnPrimeTradePL = {lnPrimeTradePL}.")
            print(f"primeR = {primeR}.")

    @staticmethod
    def getCurrentDayTime(dataFrame):
        currentDayTime = dataFrame.index[-1]
        return datetime.strptime(currentDayTime, '%Y-%m-%d %H:%M:%S')

    def __init__(self, env, n, initInvest=5600*5, eta=0.05, gamma=0.95,
                 epsilon=0.1, initType="zeros", rewardType="shapeRatio",
                 basisFctType="hypTanh123", typeFeatureVector="block",
                 tradeRandEpsilon=False, lrScheduler=False, verbose=False,
                 seed=0):

        # agent's variables
        self.env = env
        self.n = n                          # conjugated states
        self.initInvest = initInvest        # initial investment
        self.eta = eta                      # learning rate
        self.gamma = gamma                  # discount factor
        self.epsilon = epsilon              # epsilon for e-greedy policy
        self.initType = initType            # zeros, uniform01
        self.rewardType = rewardType        # shapeRatio, mean, sum
        self.basisFctType = basisFctType    # hypTanh123, tanh, relu, sigmoid
        self.typeFeatureVector = typeFeatureVector  # block, nonblock
        self.tradeRandEpsilon = tradeRandEpsilon
        self.lrScheduler = lrScheduler
        self.verbose = verbose

        # seeding the experiment
        if seed != 0:
            self.seed = seed
            np.random.seed(self.seed)
            tr.manual_seed(self.seed)

        self.spaceA = ["sell", "buy", "hold"]
        self.encodeA = {"sell": -1, "hold": 0, "buy": 1}
        self.spaceTradeStatus = ["short", "long", "out"]
        self.encodeTradeStatus = {"short": -1, "out": 0, "long": 1}
        self.zeroVector = tr.zeros((1, n + 1), dtype=tr.double)
        self.qValues = dict()
        self.primeR = 0

        if self.typeFeatureVector == "block":
            self.d = len(self.spaceA) * (self.n + 1)
        else:
            self.d = self.n + 1

        # init weight vector
        if initType == "zeros":
            self.w = tr.zeros((self.d, 1), dtype=tr.double)

        elif initType == "uniform01":
            self.w = tr.zeros((self.d, 1), dtype=tr.double)
            self.w = self.w.uniform_()

        else:
            raise ValueError(f"ERROR: initType {initType} not recognized!")

        # memory
        self.memory = pd.DataFrame(
            columns=['open', 'high', 'low', 'close', 'volume',
                     'tradeStatus', 'A', 'primeA', 'tau', 'tradePL',
                     'primeR']
        )
        self.memoryW = None
        self.memoryNablaQ = None
        self.deltas = []

        # counters
        self.randEpsilon = 0
        self.wasRandEpsilon = False
        self.countRandETrue = 0
        self.t = 1

        # initial variables
        self.dfS = self.env.S
        self.S = tr.from_numpy(self.dfS.values)
        self.A = 0
        self.tradeStatus, self.tradePL, self.tau = 0, 0, 0

        f = self.getFeatureVector(
            S=self.S,                           # current state
            A=self.A,                           # action t
            tradeStatus=self.tradeStatus,       # tradeStatus
            tradePL=self.tradePL                # current trade profit
        )

        self.Q = (self.w.T @ f).item()
        self.nablaQ = f
        self.histPL = []

    def getBasisVector(self, S, tradeStatus, tradePL):
        b = tr.zeros((1, self.n + 1), dtype=tr.double)

        for i in reversed(range(2, self.n + 2)):
            currentPrice = S[-i+1, 3]
            previousPrice = S[-i, 3]

            b[0, -i] = self.basisFunction(
                x=self.ln(currentPrice, previousPrice),
                basisFctType=self.basisFctType
            )

        if tradeStatus == 0:
            b[0, -1] = self.basisFunction(
                x=0,
                basisFctType=self.basisFctType
            )

        else:
            b[0, -1] = self.basisFunction(
                x=tradePL,
                basisFctType=self.basisFctType
            )

        return b

    def getFeatureVector(self, S, A, tradeStatus, tradePL):
        b = self.getBasisVector(
            S=S,
            tradeStatus=tradeStatus,
            tradePL=tradePL
        )

        if self.typeFeatureVector == "block":
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

        else:
            if tradeStatus == 0:
                b[0, -1] = self.basisFunction(
                    x=0,
                    basisFctType=self.basisFctType
                )
                return b.T

            else:
                b[0, -1] = self.basisFunction(
                    x=tradePL,
                    basisFctType=self.basisFctType
                )
                return b.T

    def epsilonGreedyPolicy(self, S, As, tradeStatus, tradePL):
        self.randEpsilon = np.random.uniform(low=0, high=1, size=None)
        if self.epsilon >= self.randEpsilon:
            self.wasRandEpsilon = True
            self.countRandETrue += 1
            a = np.random.choice(
                As,
                size=None,
                replace=False,
                p=None
            )
            f = self.getFeatureVector(
                S=S,                                # current state
                A=a,                                # action t
                tradeStatus=tradeStatus,            # tradeStatus
                tradePL=tradePL                     # current trade PL
            )
            q = (self.w.T @ f).item()
            return a, q, f

        else:
            self.wasRandEpsilon = False
            Q, nablaQ = {}, {}
            for a in As:
                f = self.getFeatureVector(
                    S=S,                            # current state
                    A=a,                            # action t
                    tradeStatus=tradeStatus,        # tradeStatus
                    tradePL=tradePL                 # current trade PL
                )
                Q[a] = (self.w.T @ f).item()
                nablaQ[a] = f

            # checking equal Q values for different actions.
            equalQs = {k: v for k, v in Q.items()
                       if list(Q.values()).count(v) > 1}

            # if equalQ is detected, do not trade, i.e., select action 0.
            argmax = max(Q, key=Q.get) if len(equalQs) <= 1 else 0

            return argmax, Q[argmax], nablaQ[argmax]

    def spaceAs(self, tradeStatus):
        if tradeStatus == -1:
            return [0, 1]

        elif tradeStatus == 0:
            return [-1, 0, 1]

        elif tradeStatus == 1:
            return [0, -1]

    def saveMemory(self, dfS, tradeStatus, A, primeA, tau, tradePL,
                   primeR, nablaQ):
        col = dfS.columns.to_list()
        extCols = ["tradeStatus", "A", "primeA", "tau", "tradePL", "primeR"]
        keys = col+extCols

        val1 = [dfS[k][-1] for k in dfS.keys().to_list()]
        val2 = [tradeStatus, A, primeA, tau, tradePL, primeR]
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
        self.env.runNext(A=self.A)
        dfPrime = self.env.S
        Sprime = tr.from_numpy(dfPrime.values)

        Rprime = self.rewardFunction(
            Gtplus1=self.env.histRprime,
            rewardType=self.rewardType
        )

        Aprime, Qprime, nablaQprime = self.epsilonGreedyPolicy(
            S=Sprime,
            As=self.spaceAs(self.env.tradeStatus),
            tradeStatus=self.env.tradeStatus,
            tradePL=self.env.lnTradePL
        )

        # compute TD-error
        delta = Rprime + self.gamma * Qprime - self.Q

        # reducing learning rate
        if self.lrScheduler is True:
            if (len(self.deltas) >= 10) and \
                    (self.deltas[-1] > self.deltas[-9]) and \
                    (self.deltas[-1] > self.deltas[-8]) and \
                    (self.deltas[-1] > self.deltas[-7]) and \
                    (self.deltas[-1] > self.deltas[-6]) and \
                    (self.deltas[-1] > self.deltas[-5]) and \
                    (self.deltas[-1] > self.deltas[-4]) and \
                    (self.deltas[-1] > self.deltas[-3]) and \
                    (self.deltas[-1] > self.deltas[-2]):
                self.eta /= 2

        self.w += self.eta * delta * self.nablaQ         # weight update

        self.saveMemory(
            dfS=dfPrime,
            tradeStatus=self.env.tradeStatus,
            A=self.A,
            primeA=Aprime,
            tau=len(self.env.tradeMemory["currentTradePLs"])-1,
            tradePL=self.env.tradePL,
            primeR=Rprime,
            nablaQ=self.nablaQ
        )

        self.dfS = dfPrime
        self.S = Sprime
        self.A = Aprime
        self.Q, self.nablaQ = Qprime, nablaQprime
        self.deltas.append(delta)
        self.t += 1


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = 5
    fileName = "data/WING22/WING22_30min_OLHCV.csv"
    initInvest = 56000*5

    pls = []
    for seed in range(1, 21):
        env = Environment(
            n=n,
            fileName=fileName,
            initInvest=initInvest
        )

        agent = Agent(
            env=env,
            n=n,
            initInvest=initInvest,
            eta=0.01,
            gamma=0.95,
            epsilon=0.1,
            initType="uniform01",
            rewardType="mean",
            basisFctType="hypTanh",
            typeFeatureVector="block",
            lrScheduler=True,
            verbose=False,
            seed=seed,
        )

        while env.terminal is not True:
            agent.run()

        pls.append(sum(env.histTradePLs))
        plt.plot([0]+[sum(env.histTradePLs[:i]) for i in range(len(env.histTradePLs))])
    plt.show()

    print(np.mean(pls))

    plt.plot(abs(np.array(agent.deltas)))
    plt.show()

"""
60min
n=5
eta=0.001,
gamma=0.95,
epsilon=0.1,
initType="uniform01",
rewardType="mean",
basisFctType="hypTanh",
typeFeatureVector="block",
lrScheduler=True
2919.2105263157896
"""
