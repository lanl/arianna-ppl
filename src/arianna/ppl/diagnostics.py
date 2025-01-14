import numpy as np


def ess_kish(w: np.ndarray, log: bool = True) -> float:
    """Kish Effective Sample Size.

    Kish's effective sample size. Used for weighted samples. (e.g. importance
    sampling, sequential monte carlo, particle filters.)

    https://en.wikipedia.org/wiki/Effective_sample_size

    If log is True, then the w are log weights.
    """
    if log:
        return ess_kish(np.exp(w - np.max(w)), log=False)
    else:
        return sum(w) ** 2 / sum(w**2)
