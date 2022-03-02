# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
import torch as tr
from readData import get_instrument, create_candles


class Environment:

    def __init__(self, inFolder, outFolder, ticker, candlePeriodicity):
        self.currentState = tr.DoubleTensor([0.2, 0.3, 0.4])[:, None]
        self.nextState = tr.DoubleTensor([0.2, 0.3, 0.4, 0.5])[:, None]

    def getCurrentState(self):
        return self.currentState

    def getNextState(self, A):
        return self.nextState


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
        self.eta = eta                  # learning rate
        self.gamma = gamma              # discount factor
        self.epsilon = epsilon          # epsilon for e-greedy policy
        self.initType = initType        # init for w, b, f
        self.rewardType = rewardType    # for reward function
        self.algoType = algoType        # sarsa, ql, or g-ql

        # seeding the experiment
        if seed != 0:
            self.seed = seed
            np.random.seed(self.seed)
            tr.manual_seed(self.seed)

        # sets
        self.spaceA = [-1, 0, 1]
        self.S = 0
        self.minusA, self.A = 0, 0
        self.R, self.lineR = 0, 0
        self.Q, self.nablaQ = 0, 0
        self.zeroVector = tr.zeros((1, n+1), dtype=tr.double)

        # variables that must be checked in every run
        self.t, self.tau = 1, 0
        self.d = len(self.spaceA) * (self.n + 1)

        # init vectors
        if initType == "zeros":
            self.w = tr.zeros((self.d, 1), dtype=tr.double)
        else:
            raise ValueError("ERROR: initType not recognized!")

    def getBasisVector(self, S, minusA, lineR):
        b = tr.zeros((1, self.n + 1), dtype=tr.double)

        for i in range(1, self.n+1):
            currentPrice = S[self.t-(self.n - 1)+i]
            previousPrice = S[self.t-self.n+i]

            b[0, i-1] = self.basisFunction(
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
        if self.epsilon <= np.random.uniform(low=0, high=1, size=None):
            a = np.random.choice(
                As,
                size=None,
                replace=False,
                p=None
            )
            f = self.getFeatureVector(
                S=S,                    # current state
                A=a,                    # action t
                minusA=self.minusA,     # action t-1
                lineR=self.lineR        # current trade profit
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

    def run(self):

        if self.t == 1:
            S, A = self.env.getCurrentState(), self.A

            f = self.getFeatureVector(
                S=S,                    # current state
                A=A,                    # action t
                minusA=self.minusA,     # action t-1
                lineR=self.lineR        # current trade profit
            )
            Q = (self.w.T @ f).item()
            nablaQ = f

        else:
            S, A, Q, nablaQ = self.S, self.A, self.Q, self.nablaQ

        primeS = self.env.getNextState(A)

        primeR = self.reward(
            S=S,
            primeS=primeS
        )

        primeA, primeQ, primeNablaQ = self.epsilonGreedyPolicy(
            S=primeS,
            As=self.spaceAs()
        )

        delta = primeR + self.gamma * primeQ - Q
        self.w += self.eta * delta * nablaQ

        self.t += 1
        self.tau += 1 if self.minusA != A else 0

        self.S = primeS
        self.minusA, self.A = A, primeA
        self.Q, self.nablaQ = primeQ, primeNablaQ


if __name__ == '__main__':

    agent = Agent(
        env=Environment(),
        n=2,
        eta=0.05,
        gamma=0.95,
        epsilon=0.1,
        initType="zeros",
        rewardType="immediate",
        algoType="SARSA",
        seed=0
    )

    agent.run()