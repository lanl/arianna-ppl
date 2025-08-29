"""types.

Type aliases used throughout `arianna`
"""

from numpy import ndarray

Numeric = float | ndarray
Shape = tuple[int, ...]
State = dict[str, Numeric]


class NegativeInfinityError(Exception):
    """NegativeInfinityError.

    Raised when an unexpected negative infinity is encountered,
    particularly during model fitting.
    """

    pass


class NegativeParameterError(Exception):
    """NegativeParameterError.

    Raised when a parameters is unexpectedly assigned a negative
    value, particularly during model fitting.
    """


class InvalidBoundsError(Exception):
    """InvalidBoundsError.

    Raised when a parameter provided (usually to initialize a distribution) is
    out of bounds.
    """
