from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from numpy import ndarray
from numpy.random import Generator as RNG
from numpy.random import default_rng
from scipy.special import expit, logit

from arianna.types import InvalidBoundsError, Numeric, Shape


class Distribution(ABC):
    @cached_property
    @abstractmethod
    def event_shape(self): ...

    # TODO: Think about how to implement this.
    @cached_property
    @abstractmethod
    def batch_shape(self): ...

    @abstractmethod
    def logpdf(self, x: Numeric) -> Numeric: ...

    @abstractmethod
    def sample(
        self, sample_shape: Shape = (), rng: RNG = default_rng()
    ) -> ndarray: ...

    def pdf(self, x: Numeric) -> Numeric:
        return np.exp(self.logpdf(x))

    @cached_property
    @abstractmethod
    def mean(self) -> Numeric: ...

    @cached_property
    def std(self) -> Numeric:
        return np.sqrt(self.var)

    @cached_property
    @abstractmethod
    def var(self) -> Numeric: ...


class Continuous(Distribution):
    @abstractmethod
    def to_real(self, x: Numeric) -> Numeric: ...

    @abstractmethod
    def to_native(self, z: Numeric) -> Numeric: ...

    @abstractmethod
    def logdetjac(self, z: Numeric) -> Numeric: ...


class Discrete(Distribution): ...


class Multivariate(Distribution):
    @cached_property
    @abstractmethod
    def mean(self) -> ndarray: ...

    @cached_property
    def std(self) -> ndarray:
        return np.sqrt(self.var)

    @cached_property
    @abstractmethod
    def var(self) -> ndarray: ...


class Univariate(Distribution):
    @cached_property
    def event_shape(self) -> Shape:
        return ()

    @cached_property
    @abstractmethod
    def batch_shape(self) -> Shape: ...

    def _reshape(self, sample_shape: Shape) -> Shape:
        return sample_shape + self.batch_shape

    @abstractmethod
    def _sample(self, size: Shape, rng: RNG) -> ndarray: ...

    def sample(
        self, sample_shape: Shape = (), rng: RNG = default_rng()
    ) -> ndarray:
        shape = self._reshape(sample_shape)
        return self._sample(shape, rng)


class UnivariateContinuous(Univariate, Continuous):
    def logpdf_plus_logdetjac(self, z: Numeric) -> Numeric:
        """Logpdf plus the log absolute determinant of the jacobian.

        Logpdf plus the log absolute determinant of the jacobian, evaluated at
        parameter on the transformed (real) space.
        """
        x = self.to_native(z)
        return self.logpdf(x) + self.logdetjac(z)

    @abstractmethod
    def logcdf(self, x: Numeric) -> Numeric: ...

    def cdf(self, x: Numeric) -> Numeric:
        with np.errstate(divide="ignore"):
            return np.exp(self.logcdf(x))

    def survival(self, x: Numeric) -> Numeric:
        return 1 - self.cdf(x)

    def logsurvival(self, x: Numeric) -> Numeric:
        return np.log1p(-self.cdf(x))


class Positive(UnivariateContinuous):
    @abstractmethod
    def _logpdf(self, x: Numeric) -> Numeric: ...

    def to_real(self, x: Numeric) -> Numeric:
        return np.log(x)

    def to_native(self, z: Numeric) -> Numeric:
        return np.exp(z)

    def logdetjac(self, z: Numeric) -> Numeric:
        return z

    def logpdf(self, x: Numeric) -> Numeric:
        # ignore divide by zero encountered in log.
        with np.errstate(divide="ignore"):
            return np.where(x > 0, self._logpdf(np.maximum(0, x)), -np.inf)


class LowerUpperBounded(UnivariateContinuous, ABC):
    @abstractmethod
    def _logpdf(self, x: Numeric) -> Numeric: ...

    def __init__(self, lower: Numeric, upper: Numeric, check: bool = True):
        self.lower = lower
        self.upper = upper
        self.range = self.upper - self.lower
        if check and np.any(self.range <= 0):
            raise InvalidBoundsError(
                "In LowerUpperBounded, lower bound needs to be strictly less than upper bound!"
            )

    def to_real(self, x: Numeric) -> Numeric:
        return logit((x - self.lower) / self.range)

    def to_native(self, z: Numeric) -> Numeric:
        return expit(z) * self.range + self.lower

    def logdetjac(self, z: Numeric) -> Numeric:
        return np.log(self.range) + z - 2 * np.logaddexp(0, z)

    def logpdf(self, x: Numeric) -> Numeric:
        # ignore divide by zero encountered in log.
        with np.errstate(divide="ignore"):
            return np.where(
                (self.lower < x) & (x < self.upper),
                self._logpdf(np.clip(x, self.lower, self.upper)),
                -np.inf,
            )


class MultivariateContinuous(Multivariate, Continuous):
    def logpdf_plus_logdetjac(self, z: ndarray) -> Numeric:
        """Logpdf plus the log absolute determinant of the jacobian.

        Logpdf plus the log absolute determinant of the jacobian, evaluated at
        parameter on the transformed (real) space.
        """
        x = self.to_native(z)
        return self.logpdf(x) + self.logdetjac(z)

    @abstractmethod
    def cov(self) -> ndarray: ...

    @abstractmethod
    def mean(self) -> ndarray: ...


class Real:
    def to_real(self, x: Numeric) -> Numeric:
        return x

    def to_native(self, z: Numeric) -> Numeric:
        return z

    def logdetjac(self, z: Numeric) -> Numeric:
        return 0


class UnivariateReal(Real, UnivariateContinuous): ...


class MultivariateReal(Real, MultivariateContinuous): ...
