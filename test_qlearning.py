import numpy as np
import torch as tr
import Qlearning
import pytest

"""To run, please use: pytest test_qlearning.py -v"""

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


def test_agent_init():
    """ Testing the initialization of the class Agent:

    1) testing if weight vector w is a column vector.
    """
    agent = Qlearning.Agent(
        N=3,
        initType="test",
        rewardType="shapeRatio",
        seed=1
    )

    # 1st test: if weight vector w is a column vector.
    assert agent.w.size()[1] == 1


def test_agent_featureGeneration():
    """Testing the featureGeneration method from Agent's class:

     1) testing if the feature vector phi is a column vector.
     2) testing if the feature vector phi returns the correct computations
     for a determined scenario.
     """
    agent = Qlearning.Agent(
        N=3,
        initType="test",
        rewardType="shapeRatio",
        seed=1
    )

    agent.run(
        Rtplus1=0,
        Stplus1=database[:4, 0].reshape(-1, 1),
        tradingExe=False
    )

    # 1st test: if the feature vector phi is a column vector.
    assert agent.w.size()[1] == 1

    # 2nd test: if the feature vector phi returns the correct computations
    # for a determined scenario.
    epsilon = 0.0001

    assert (agent.phi[0, 0].item() - 1.0) < epsilon
    assert (agent.phi[1, 0].item() - 0.024692589417099953) < epsilon
    assert (agent.phi[2, 0].item() - 0.02409752830862999) < epsilon
    assert (agent.phi[3, 0].item() - 0.06899283826351166) < epsilon


def test_agent_basis():
    """ Testing the basis method from Agent's class:

    1) testing the following cases:
        basis(0.5).item()
        >> 3

        basis(0).item()
        >> 2

        basis(-0.5).item()
        >> 1
    """
    agent = Qlearning.Agent(
        N=3,
        initType="test",
        rewardType="shapeRatio",
        seed=1
    )

    assert agent.basis(x=0.5) == 3
    assert agent.basis(x=0.) == 2
    assert agent.basis(x=-0.5) == 1
