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
        return tr.log(currentPrice / previousPrice)

    @staticmethod
    def basis(x: float, a: float = 2, b: float = 1, c: float = 10 ** 15,
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

    def __init__(self, N, eta=0.05, gamma=0.95, epsilon=0.1,
                 initType="uniform", rewardType="shapeRatio", seed=0):

        # agent's parameters
        self.N = N
        self.eta = eta                      # learning rate
        self.gamma = gamma                  # discount factor
        self.epsilon = epsilon              # epsilon for e-greedy policy
        self.initType = initType
        self.rewardType = rewardType
        self.seed = seed

        # seeding the experiment
        if seed != 0:
            tr.manual_seed(self.seed)

        # variables that must be checked in every run
        self.tradingStatus = 0
        self.t = 0

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

        # variables for action-space and Q-value
        self.A = {0, 1, 2}
        self.Q = tr.zeros(1, 3, dtype=tr.float32)

        # A_{t} = \rho
        self.At = 0

    def computeQvalue(self):
        for a in self.A:
            self.phi[-1, 0] = a

            for i in range(self.w.shape[0]):
                self.Q[0, a] += \
                    self.w[i, 0].item() * self.basis(x=self.phi[i, 0].item())

    def eGreedyPolicy(self):
        if np.random.uniform(0, 1, 1) <= self.epsilon:
            rho = np.random.choice([0, 1, 2])
        else:
            rho = tr.argmax(self.Q).item()

        if self.tradingStatus == 1:  # long
            # not allowing to double position
            rho = 0 if rho == 1 else rho

        elif self.tradingStatus == 2:  # short
            # not allowing to double position
            rho = 0 if rho == 2 else rho

        return rho

    def run(self, Stplus1: tr.Tensor, wasExecuted: bool) -> int:
        # checking if Stplus1 was correctly given...
        if not isinstance(Stplus1, tr.Tensor):
            raise ValueError("ERROR: Stplus1 variable must be tensor.")

        # computing gradient step...
        if self.t != 0:
            # updating trading status...
            if wasExecuted:                                  # TODO: fix
                self.tradingStatus = self.At

            # updating reward and return...
            # TODO: this step

            # updating weights...
            # TODO: this step

        # computing features...
        for i in range(1, self.N + 1):
            self.phi[i, 0] = self.featureGeneration(
                previousPrice=Stplus1[i - 1, 0],
                currentPrice=Stplus1[i, 0]
            )

        # computing q-value
        self.computeQvalue()

        # e-greedy policy
        self.At = self.eGreedyPolicy()

        # updating time step
        self.t += 1

        # returning the chosen action
        return self.At


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
        Stplus1=database[:4, 0].unsqueeze(dim=1),
        wasExecuted=False
    )

    # rho as input to the environment
    Stplus1 = database[:5, 0].unsqueeze(dim=1)
    tradingExe = False if rho == 0 else True  # TODO else option
    # Stplus1 and wasTradingExe as output from the environment

    rho = agent.run(
        Stplus1=Stplus1,
        wasExecuted=tradingExe
    )
