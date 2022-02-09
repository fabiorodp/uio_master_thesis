import numpy as np
import torch as tr

# tr.cuda.is_available()
# tr.cuda.current_device()
# tr.cuda.device(0)
# tr.cuda.device_count()
# tr.cuda.get_device_name(0)


def e(close_t: float, close_t_minus_1: float) -> tr.Tensor:
    """
    e(10.75, 11).item()
    >> -0.02298949658870697
    """
    return tr.log(tr.tensor(close_t)/tr.tensor(close_t_minus_1))


def phi(x: tr.FloatTensor, a: int = 2, b: int = 1, c: float = 10**15, d=-1) \
        -> tr.FloatTensor:
    """
    phi(0.5).item()
    >> 3

    phi(0).item()
    >> 2

    phi(-0.5).item()
    >> 1
    """
    cx = - tr.FloatTensor([c]) * x
    denum = tr.FloatTensor([1]) + tr.FloatTensor([b]) * tr.exp(cx)
    return (tr.FloatTensor([a]) / denum) - tr.FloatTensor([d])


def getSt(priceIn: list, atminus1: float, N: int) -> tr.Tensor:
    # st_list = [1.0]
    st_list = [e(priceIn[-i], priceIn[-i-1]).item()
               for i in reversed(range(1, N+1))]
    st_list += [atminus1]
    return tr.FloatTensor(st_list)


def init_wk(stts: tr.Tensor, initType: str = "uniform") -> tr.Tensor:
    """
    Initializing weight vector wk.
    """
    if initType == "uniform":
        return tr.FloatTensor([stts.shape[0]+1]).uniform_(-1, 1)
    else:
        raise ValueError(f"ERROR: initType {initType} not recognized...")


def actionFunction(status: tr.Tensor) -> tr.Tensor:
    """
    Checking last action and returning the possible actions.
    """
    if status[-1].item() == 0.:
        return tr.IntTensor([-1, 0, 1])

    elif status[-1].item() == -1.:
        return tr.IntTensor([0, 1])

    elif status[-1].item() == 1.:
        return tr.IntTensor([-1, 0])

    else:
        raise ValueError(f"ERROR: Status {status[-1].item()} "
                         f"not recognized...")


def estimateQ(St: tr.FloatTensor, at: tr.FloatTensor, wk: tr.FloatTensor) \
        -> tr.FloatTensor:
    r"""
    Calculating the state-action value.

    Formula:
    ~~~~~~~~~~~~~~~~~~~~
    Q(S_{t}, a_{t}, \overrightarrow{w}_{k}) = w_{k_{0}}
    + \sum_{n=1}^{N} w_{k_{n}} phi(s_{t_{n}}) + w_{k_{N}} phi(a_{t})
    """
    qt = wk[0].item()

    for idx, s in enumerate(St):
        qt += wk[idx+1].item() * phi(x=s).item()

    qt += wk[-1].item() * phi(x=at).item()

    return tr.FloatTensor([qt])


if __name__ == '__main__':
    # initializing weight vector wk
    wk = tr.FloatTensor([-0.2, 0.2, 0.4, -0.4, -0.1, 0.1])

    # initializing state vector St
    close = [10.00, 10.25, 10.50, 11.25]  # , 11.75, 12.00, 11.00, 10.75]
    St = getSt(close, 0, 3)

    # checking last action and returning the possible actions
    actionSpace = actionFunction(status=St)

    qt = []
    for at in actionSpace:
        qt.append(estimateQ(St, at, wk))
