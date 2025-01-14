from numpy import ndarray

Numeric = float | ndarray
Shape = tuple[int, ...]
State = dict[str, Numeric]


class NegativeInfinityError(Exception):
    pass


class NegativeParameterError(Exception): ...


class InvalidBoundsError(Exception): ...
