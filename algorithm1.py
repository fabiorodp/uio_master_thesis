# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
import torch as tr
import pandas as pd

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

    def __init__(self, env, n, eta=0.05, gamma=0.95, epsilon=0.1,
                 initType="zeros", rewardType="shapeRatio",
                 basisFctType="hypTanh123", typeFeatureVector="block",
                 tradeRandEpsilon=False, verbose=False, seed=0):

        # agent's parameters
        self.env = env
        self.n = n                          # conjugated states
        self.eta = eta                      # learning rate
        self.gamma = gamma                  # discount factor
        self.epsilon = epsilon              # epsilon for e-greedy policy
        self.initType = initType            # zeros, uniform01
        self.rewardType = rewardType        # shapeRatio, mean, sum
        self.basisFctType = basisFctType    # hypTanh123, hypTanh, relu, sigmoid
        self.typeFeatureVector = typeFeatureVector  # block, nonblock
        self.tradeRandEpsilon = tradeRandEpsilon
        self.verbose = verbose

        # seeding the experiment
        if seed != 0:
            self.seed = seed
            np.random.seed(self.seed)
            tr.manual_seed(self.seed)

        # variables
        self.dfS = None
        self.spaceA = [-1, 0, 1]
        self.S, self.A = 0, 0
        self.tradeStatus, self.tradePL, self.tau = 0, 0, 0
        self.primeR = 0
        self.Q, self.nablaQ = 0, 0
        self.zeroVector = tr.zeros((1, n + 1), dtype=tr.double)
        self.randEpsilon = 0
        self.t = 1
        self.deltas = []
        self.wasRandEpsilon = False

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
                     "entryPrice", "tradeStatus", "A", "tradePL", "tau",
                     "primeR", "wasRandEpsilon"]
        )
        self.memoryW = None
        self.memoryNablaQ = None

        # counters
        self.countRandETrue = 0

    def getBasisVector(self, S, tradeStatus, tradePL):
        b = tr.zeros((1, self.n + 1), dtype=tr.double)

        for i in reversed(range(2, self.n + 2)):
            currentPrice = S[-i + 1]
            previousPrice = S[-i]

            b[0, i - 1] = self.basisFunction(
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
            Q, nablaQ = {}, {}
            self.wasRandEpsilon = False
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
            return [-1, 0]

    def saveTapeScript(self, dfS, entryPrice, tradeStatus, A, tradePL, tau,
                       primeR, wasRandEpsilon):
        col = dfS.columns.to_list()
        extCols = ["entryPrice", "tradeStatus", "A", "tradePL", "tau",
                   "primeR", "wasRandEpsilon"]
        keys = col + extCols

        val1 = [dfS[k][-1] for k in dfS.keys().to_list()]
        val2 = [entryPrice, tradeStatus, A, tradePL, tau, primeR,
                wasRandEpsilon]
        vals = val1 + val2

        timeIdx = dfS.index.to_list()[-1]

        memory = pd.DataFrame(vals).T
        memory.columns = keys
        memory.index = [timeIdx]
        self.memory = pd.concat([self.memory, memory], axis=0)

    def saveMemory(self, nablaQ):

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
            dfS = self.env.getCurrentState()
            self.S = tr.from_numpy(dfS.values[:, 3])[:, None]

            f = self.getFeatureVector(
                S=self.S,                           # current state
                A=self.A,                           # action t
                tradeStatus=self.tradeStatus,       # tradeStatus
                tradePL=self.tradePL                # current trade profit
            )

            self.Q = (self.w.T @ f).item()
            self.nablaQ = f

        self.t += 1

        if self.tradeRandEpsilon is False:
            (dfPrimeS, _, entryPrice, primeTradePL, primeTau,
             primeTradeStatus, lnPrimeTradePL) = \
                self.env.getNextState(self.A, self.wasRandEpsilon)

        else:
            (dfPrimeS, _, entryPrice, primeTradePL, primeTau,
             primeTradeStatus, lnPrimeTradePL) = \
                self.env.getNextState(self.A, False)

        primeS = tr.from_numpy(dfPrimeS.values[:, 3])[:, None]

        primeR = self.rewardFunction(
            Gtplus1=self.env.histR,
            rewardType=self.rewardType
        )

        self.saveTapeScript(
            dfS=dfPrimeS,
            entryPrice=int(entryPrice),
            tradeStatus=primeTradeStatus,
            A=self.A,
            tradePL=primeTradePL,
            tau=primeTau,
            primeR=primeR,
            wasRandEpsilon=self.wasRandEpsilon
        )

        primeA, primeQ, primeNablaQ = self.epsilonGreedyPolicy(
            S=primeS,
            As=self.spaceAs(tradeStatus=primeTradeStatus),
            tradeStatus=primeTradeStatus,
            tradePL=lnPrimeTradePL
        )

        # compute TD-error
        delta = primeR + self.gamma * primeQ - self.Q

        # reducing learning rate
        """if (len(self.deltas) >= 5) and (self.deltas[-1] > self.deltas[-3]):
            eta = self.eta / 2
        else:"""
        eta = self.eta

        self.w += eta * delta * self.nablaQ         # weight

        if self.verbose is True:
            print(f"\nThe prime time is {dfPrimeS.index.to_list()[-1]}.")
            print(f"The prime closed price is {dfPrimeS.iloc[-1, 3]}.")
            print(f"\ntradeStatus = {self.tradeStatus}.")
            print(f"primeTradeStatus = {primeTradeStatus}.")
            print(f"A = {self.A}.")
            print(f"primeA = {primeA}.")
            print(f"tau = {self.tau}.")
            print(f"primeTau = {primeTau}.")
            print(f"entryPrice = {entryPrice}.")
            print(f"primeTradePL = {primeTradePL}.")
            print(f"lnPrimeTradePL = {lnPrimeTradePL}.")
            print(f"primeR = {primeR}.")

        self.saveMemory(nablaQ=self.nablaQ)

        self.S = primeS
        self.A = primeA
        self.Q, self.nablaQ = primeQ, primeNablaQ
        self.deltas.append(delta)
        self.tradeStatus = primeTradeStatus
        self.tradePL = primeTradePL
        self.tau = primeTau
        self.eta = eta


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    from environment3a import Environment

    n = 10  # 2 and 5 works for long range. 60 works for short range.
    fileName = "data/WING22/WING22_30min_OLHCV.csv"
    # fileName = "data/WING22/CSV/day3/WING22_5min_OLHCV.csv"

    env = Environment(
        n=n,
        fileName=fileName
    )

    agent = Agent(
        env=env,
        n=n,
        eta=0.001,      # 0.005 for n=30 and 30min or 0.05 for n=5 and 1min
        gamma=0.9,
        epsilon=0.15,   # 0.05 for n=30 and 30min or 0.05 for n=5 and 1min
        initType="uniform01",
        rewardType="shapeRatio",  # mean for n=30 and 30min or for n=5 and 1min
        basisFctType="sigmoid",
        typeFeatureVector="block",
        tradeRandEpsilon=False,
        verbose=True,
        seed=20,                        # 5, 50, 20 works, 2 not work
    )

    while env.terminal is not True:
        agent.run()

    # fixing agent.memory
    for i in range(len(agent.memory)):
        if agent.memory["tradeStatus"][i] == -1:
            agent.memory["tradePL"][i] = agent.memory["entryPrice"][i] - \
                                         agent.memory["close"][i]
        elif agent.memory["tradeStatus"][i] == 1:
            agent.memory["tradePL"][i] = agent.memory["entryPrice"][i] + \
                                         agent.memory["close"][i]
        elif agent.memory["tradeStatus"][i] == 0:
            agent.memory["tradePL"][i] = 0

    cumulativeReturn = []
    taus = []
    for i in range(len(agent.memory)):
        if (agent.memory["tradeStatus"][i] == 0) and (
                agent.memory["A"][i] == -1):
            cumulativeReturn.append(agent.memory["tradePL"][i - 1])
            taus.append(agent.memory["tau"][i - 1])

        elif (agent.memory["tradeStatus"][i] == 0) and (
                agent.memory["A"][i] == 1):
            cumulativeReturn.append(agent.memory["tradePL"][i - 1])
            taus.append(agent.memory["tau"][i - 1])

    if agent.memory["tau"][-1] != 0:
        cumulativeReturn.append(agent.memory["tradePL"][-1])
        taus.append(agent.memory["tau"][1])

    axisY = [0]+[sum(cumulativeReturn[:i]) for i in range(1, len(cumulativeReturn)+1)]
    axisX = [i for i in range(len(axisY))]

    sns.lineplot(x=axisX, y=axisY)
    plt.show()

    plt.plot(agent.deltas)
    plt.show()

    cumulativeReturn = [e for e in cumulativeReturn if e != 0]
    sns.distplot(cumulativeReturn)
    plt.show()

    sum(cumulativeReturn)

    sum(env.portfolioPLs)

    axisY = [0] + [sum(agent.memory["primeR"][:i]) for i in
                   range(1, len(agent.memory["primeR"]) + 1)]
    axisX = [i for i in range(len(agent.memory["primeR"].values))]

    sns.lineplot(x=axisX, y=agent.memory["primeR"].values)
    plt.show()
