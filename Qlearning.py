import numpy as np
import torch as tr

# tr.cuda.is_available()
# tr.cuda.current_device()
# tr.cuda.device(0)
# tr.cuda.device_count()
# tr.cuda.get_device_name(0)


class Agent:

    @staticmethod
    def init_wk(N: int, initType: str = "uniform") -> tr.Tensor:
        """
        Initializing weight vector wk...
        """
        if initType == "uniform":
            return tr.Tensor(N + 2).uniform_(-1).unsqueeze(dim=1)
        elif initType == "test":
            return tr.Tensor([-0.2, 0.2, 0.4, -0.4, -0.1]).unsqueeze(dim=1)
        else:
            raise ValueError(f"ERROR: initType {initType} not recognized...")

    @staticmethod
    def featureGeneration(previousPrice: tr.Tensor,
                          currentPrice: tr.Tensor) -> tr.Tensor:
        """Computing a feature..."""
        return tr.log(currentPrice/previousPrice)

    @staticmethod
    def basis(x: float, a: float = 2, b: float = 1, c: float = 10**15,
              d: float = -1) -> float:
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

    def __init__(self, N, initType="uniform", rewardType="shapeRatio",
                 seed=0):

        # agent's parameters
        self.N = N
        self.initType = initType
        self.rewardType = rewardType
        self.seed = seed

        # seeding the experiment
        if seed != 0:
            tr.manual_seed(self.seed)

        # initialization of a column vector for weights
        self.w = self.init_wk(
            N=self.N,
            initType=self.initType
        )

        # initialization of a column vector for features
        # alternative for .unsqueeze(dim=1) = [:, None] = .reshape(-1, 1)
        self.phi = tr.Tensor(
            [1.0] + [0.0 for _ in range(self.N + 1)]
        ).unsqueeze(dim=1)

        self.numEpisodes = 0

        self.A = {0, 1, 2}
        self.tradingStatus = 0
        self.Q = tr.zeros(1, 3, dtype=tr.float32)
        self.At = 0

        self.St = tr.FloatTensor(self.N)
        self.Stplus1 = tr.FloatTensor(self.N)

        self.Rtplus1 = 0
        self.Gtplus1 = 0
        self.Atminus1 = 0

    def run(self, Rtplus1, Stplus1: tr.Tensor, tradingExe: bool) -> int:
        if not isinstance(Stplus1, tr.Tensor):
            raise ValueError("ERROR: Stplus1 variable must be tensor.")

        if tradingExe:  # fix
            self.tradingStatus = self.At

        # updating weights step...
        if self.numEpisodes != 0:
            pass

        # computing the features
        for i in range(1, self.N + 1):
            self.phi[i, 0] = self.featureGeneration(
                previousPrice=Stplus1[i-1, 0],
                currentPrice=Stplus1[i, 0]
            )

        # computing q-value
        for a in self.A:
            self.phi[-1, 0] = a

            for i in range(self.w.shape[0]):
                self.Q[0, a] += \
                    self.w[i, 0].item() * self.basis(x=self.phi[i, 0].item())

        # e-greedy step...
        # not allowing the algorithm to double position
        rho = 0  # do nothing as default
        if self.tradingStatus == 0:    # neutron
            rho = tr.argmax(self.Q).item()

        elif self.tradingStatus == 1:  # long
            if tr.argmax(self.Q).item() == 1:
                rho = 0
            else:
                rho = tr.argmax(self.Q).item()

        elif self.tradingStatus == 2:  # short
            if tr.argmax(self.Q).item() == 2:
                rho = 0
            else:
                rho = tr.argmax(self.Q).item()

        self.At = rho
        self.numEpisodes += 1

        return rho


class Environment:
    def __init__(self, st, at):
        self.st = st
        self.at = at


if __name__ == '__main__':

    database = tr.Tensor(
        [
            [10.00],
            [10.25],
            [10.50],
            [11.25],
            [11.75],
            [12.00],
            [11.00],
            [10.75]]
    )

    agent = Agent(
        N=3,
        initType="test",
        rewardType="shapeRatio"
    )

    rho = agent.run(
        Rtplus1=0,
        Stplus1=database[:4, 0].unsqueeze(dim=1),
        tradingExe=False
    )
