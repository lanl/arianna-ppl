from functools import cached_property
from typing import Optional

import numpy as np
from numpy import ndarray
from numpy.random import Generator as RNG
from numpy.random import default_rng
from scipy.special import (
    betainc,
    betaln,
    gammaln,
    gdtr,
    gdtrc,
    log_ndtr,
    ndtr,
)

from arianna.types import NegativeParameterError

from .abstract import (
    Distribution,
    LowerUpperBounded,
    MultivariateContinuous,
    MultivariateReal,
    Numeric,
    Positive,
    Shape,
    UnivariateReal,
)


class IndependentRagged(Distribution): ...


class Independent(Distribution):
    def __init__(self, dists: list[Distribution]):
        assert self.is_same_family(dists)
        self.dists = dists

    def is_same_family(self, dists: list[Distribution]) -> bool:
        first_type = type(dists[0])
        return all(type(d) is first_type for d in dists)

    def logpdf(self, x: list[Numeric]) -> Numeric:
        return sum(di.logpdf(xi) for di, xi in zip(self.dists, x))

    def sample(self, sample_shape=[]) -> ndarray:
        # TODO: Check the logic.
        return np.stack([di.sample(sample_shape) for di in self.dists])


class Uniform(LowerUpperBounded):
    @classmethod
    def from_mean_shift(cls, mean, shift):
        return cls(mean - shift, mean + shift)

    @cached_property
    def batch_shape(self) -> Shape:
        return np.broadcast_shapes(np.shape(self.lower), np.shape(self.upper))

    def _logpdf(self, x: Numeric) -> Numeric:
        return -np.log(self.range)

    def logcdf(self, x: Numeric) -> Numeric:
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(x))

    def cdf(self, x: Numeric) -> Numeric:
        return np.clip((x - self.lower) / self.range, 0, 1)

    def _sample(self, size: Shape, rng: RNG) -> ndarray:
        return rng.uniform(self.lower, self.upper, size=size)

    @cached_property
    def mode(self) -> Numeric:
        # FIXME: Really, it should be anything in [lower, upper].
        return self.mean

    @cached_property
    def median(self) -> Numeric:
        return self.mean

    @cached_property
    def mean(self) -> Numeric:
        return (self.lower + self.upper) / 2

    @cached_property
    def var(self) -> Numeric:
        return np.square(self.range) / 12


class Beta(LowerUpperBounded):
    def __init__(self, a: Numeric, b: Numeric, check: bool = True):
        super().__init__(lower=0, upper=1)
        if check and np.any(a < 0):
            raise NegativeParameterError(
                "In Beta(a,b), `a` must be strictly positive!"
            )
        if check and np.any(b < 0):
            raise NegativeParameterError(
                "In Beta(a,b), `b` must be strictly positive!"
            )
        self.a = a
        self.b = b

    @cached_property
    def mode(self) -> Numeric:
        # https://en.wikipedia.org/wiki/Beta_distribution
        raise NotImplementedError

    @cached_property
    def batch_shape(self) -> Shape:
        return np.broadcast(self.a, self.b).shape

    def _logpdf(self, x: Numeric) -> Numeric:
        # Hide warnings if 0 * inf, which is nan. Should just return -inf.
        with np.errstate(invalid="ignore"):
            return (
                (self.a - 1) * np.log(x)
                + (self.b - 1) * np.log1p(-x)
                - betaln(self.a, self.b)
            )

    def logcdf(self, x: Numeric) -> Numeric:
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(x))

    def cdf(self, x: Numeric) -> Numeric:
        return betainc(self.a, self.b, np.clip(x, 0, 1))

    def _sample(self, size: Shape, rng: RNG) -> ndarray:
        return rng.beta(self.a, self.b, size=size)

    @cached_property
    def mean(self) -> Numeric:
        return self.a / (self.a + self.b)

    @cached_property
    def var(self):
        c = self.a + self.b
        return self.a * self.b / (np.square(c) * (c + 1))


# TODO (12/4/2024): Write tests for all methods in Scaled Beta.
class ScaledBeta(LowerUpperBounded):
    def __init__(
        self,
        a: Numeric,
        b: Numeric,
        lower: Numeric,
        upper: Numeric,
        check: bool = True,
    ):
        super().__init__(lower=lower, upper=upper)
        if check and np.any(a < 0):
            raise NegativeParameterError(
                "In ScaledBeta(a,b,lower,upper), `a` must be strictly positive!"
            )
        if check and np.any(b < 0):
            raise NegativeParameterError(
                "In ScaledBeta(a,b,lower,upper), `b` must be strictly positive!"
            )

        self.a = a
        self.b = b
        self.base_dist = Beta(self.a, self.b, check=False)

    @cached_property
    def batch_shape(self) -> Shape:
        return np.broadcast(self.a, self.b, self.lower, self.upper).shape

    def _broadcast(self, x: Numeric) -> Numeric:
        shape = np.broadcast_shapes(np.shape(x), self.batch_shape)
        return np.broadcast_to(x, shape)

    def _to_unit_interval(self, x: Numeric) -> Numeric:
        return self._broadcast((x - self.lower) / self.range)

    def _from_unit_interval(self, y: Numeric) -> Numeric:
        return self._broadcast(y * self.range + self.lower)

    def _sample(self, size: Shape, rng: RNG) -> Numeric:
        return self._from_unit_interval(
            self.base_dist._sample(size=size, rng=rng)
        )

    def cdf(self, x: Numeric) -> Numeric:
        return self.base_dist.cdf(self._to_unit_interval(x))

    def logcdf(self, x: Numeric) -> Numeric:
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(x))

    def _logpdf(self, x: Numeric) -> Numeric:
        return self.base_dist._logpdf(self._to_unit_interval(x)) - np.log(
            self.range
        )

    @cached_property
    def mean(self) -> Numeric:
        return self._from_unit_interval(self.base_dist.mean)

    @cached_property
    def var(self) -> Numeric:
        return self.base_dist.var * self.range**2


class Gamma(Positive):
    @classmethod
    def from_mean_std(cls, mean, std, check: bool = True):
        if check and np.any(mean < 0):
            raise NegativeParameterError(
                "In Gamma.from_mean_std(mean, std), `mean` must be strictly positive!"
            )
        if check and np.any(std < 0):
            raise NegativeParameterError(
                "In Gamma.from_mean_std(mean, std), `std` must be strictly positive!"
            )

        var = std**2
        return cls(shape=mean**2 / var, scale=var / mean)

    def __init__(self, shape: Numeric, scale: Numeric, check: bool = True):
        if check and np.any(shape < 0):
            raise NegativeParameterError(
                "In Gamma(shape, scale), `shape` must be strictly positive!"
            )
        if check and np.any(scale < 0):
            raise NegativeParameterError(
                "In Gamma(shape, scale), `scale` must be strictly positive!"
            )

        self.shape = shape
        self.scale = scale

    @cached_property
    def batch_shape(self) -> Shape:
        return np.broadcast(self.shape, self.scale).shape

    def _logpdf(self, x: Numeric) -> Numeric:
        # Hide warnings if 0 * inf, which is nan. Should just return -inf.
        with np.errstate(invalid="ignore"):
            return (
                -gammaln(self.shape)
                - self.shape * np.log(self.scale)
                + (self.shape - 1) * np.log(x)
                - x / self.scale
            )

    def logcdf(self, x: Numeric) -> Numeric:
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(x))

    def cdf(self, x: Numeric) -> Numeric:
        return gdtr(1 / self.scale, self.shape, np.maximum(0, x))

    def survival(self, x: Numeric) -> Numeric:
        return gdtrc(1 / self.scale, self.shape, np.maximum(0, x))

    def _sample(self, size: Shape, rng: RNG) -> ndarray:
        return rng.gamma(shape=self.shape, scale=self.scale, size=size)

    @cached_property
    def mean(self) -> Numeric:
        return self.shape * self.scale

    @cached_property
    def var(self) -> Numeric:
        return self.shape * np.square(self.scale)

    @cached_property
    def mode(self) -> Numeric:
        return np.where(self.shape > 1, self.scale * (self.shape - 1), 0.0)


# https://en.wikipedia.org/wiki/Inverse-gamma_distribution
class InverseGamma(Positive):
    @classmethod
    def from_mean_std(cls, mean, std, check: bool = True):
        if check and np.any(mean < 0):
            raise NegativeParameterError(
                "In InverseGamma.from_mean_std(mean, mean), `mean` must be strictly positive!"
            )
        if check and np.any(std < 0):
            raise NegativeParameterError(
                "In InverseGamma.from_mean_std(mean, mean), `std` must be strictly positive!"
            )

        shape = (mean / std) ** 2 + 2
        scale = mean * (shape - 1)
        return cls(shape, scale)

    def __init__(self, shape: Numeric, scale: Numeric, check: bool = True):
        if check and np.any(shape < 0):
            raise NegativeParameterError(
                "In InverseGamma(shape, scale), `shape` must be strictly positive!"
            )
        if check and np.any(scale < 0):
            raise NegativeParameterError(
                "In InverseGamma(shape, scale), `scale` must be strictly positive!"
            )

        self.shape = shape
        self.scale = scale

    @cached_property
    def batch_shape(self) -> Shape:
        return np.broadcast(self.shape, self.scale).shape

    @cached_property
    def mean(self) -> Numeric:
        return np.where(self.shape > 1, self.scale / (self.shape - 1), np.nan)

    @cached_property
    def var(self) -> Numeric:
        return np.where(self.shape > 2, self.mean**2 / (self.shape - 2), np.nan)

    @cached_property
    def mode(self) -> Numeric:
        return self.scale / (self.shape + 1)

    def _logpdf(self, x: Numeric) -> Numeric:
        # Hide warnings if 0 * inf, which is nan. Should just return -inf.
        with np.errstate(invalid="ignore", divide="ignore"):
            return (
                self.shape * np.log(self.scale)
                - gammaln(self.shape)
                - (self.shape + 1) * np.log(x)
                - self.scale / x
            )

    def logcdf(self, x: Numeric) -> Numeric:
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(x))

    def cdf(self, x: Numeric) -> Numeric:
        with np.errstate(divide="ignore"):
            x = np.maximum(0, x)
            return gdtrc(self.scale, self.shape, 1 / x)

    def _sample(self, size: Shape, rng: RNG) -> ndarray:
        return 1 / rng.gamma(shape=self.shape, scale=1 / self.scale, size=size)


class LogNormal(Positive):
    @classmethod
    def from_mean_std(cls, mean, std, check: bool = True):
        if check and np.any(mean < 0):
            raise NegativeParameterError(
                "In LogNormal.from_mean_std(mean, std), `mean` must be strictly positive!"
            )
        if check and np.any(std < 0):
            raise NegativeParameterError(
                "In LogNormal.from_mean_std(mean, std), `std` must be strictly positive!"
            )
        var = std**2
        sigma_squared = np.log1p(var / mean**2)
        mu = np.log(mean) - sigma_squared / 2
        sigma = np.sqrt(sigma_squared)
        return cls(mu, sigma)

    def __init__(self, mu: Numeric, sigma: Numeric, check: bool = True):
        if check and np.any(sigma < 0):
            raise NegativeParameterError(
                "In LogNormal(mu, sigma), `sigma` must be strictly positive!"
            )

        self.mu = mu
        self.sigma = sigma

    @cached_property
    def batch_shape(self) -> Shape:
        return np.broadcast(self.mu, self.sigma).shape

    @cached_property
    def mean(self) -> Numeric:
        return np.exp(self.mu + self.sigma**2 / 2)

    @cached_property
    def var(self) -> Numeric:
        return (np.exp(self.sigma**2) - 1) * np.exp(2 * self.mu + self.sigma**2)

    @cached_property
    def mode(self) -> Numeric:
        return np.exp(self.mu - self.sigma**2)

    @cached_property
    def median(self) -> Numeric:
        return np.exp(self.mu)

    def _logpdf(self, x: Numeric) -> Numeric:
        # Hide warnings if 0 * inf, which is nan. Should just return -inf.
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (np.log(x) - self.mu) / self.sigma
            return -np.log(x * self.sigma * np.sqrt(2 * np.pi)) - z**2 / 2

    def logcdf(self, x: Numeric) -> Numeric:
        x = np.maximum(0, x)
        with np.errstate(divide="ignore"):
            z = (np.log(x) - self.mu) / self.sigma
            return log_ndtr(z)

    def cdf(self, x: Numeric) -> Numeric:
        x = np.maximum(0, x)
        with np.errstate(divide="ignore"):
            z = (np.log(x) - self.mu) / self.sigma
            return ndtr(z)

    def _sample(self, size: Shape, rng: RNG) -> ndarray:
        return rng.lognormal(self.mu, self.sigma, size=size)


class Weibull(Positive): ...


class Gumbel(UnivariateReal): ...


class Logistic(Positive): ...


class LogLogistic(UnivariateReal): ...


class Normal(UnivariateReal):
    def __init__(
        self, loc: Numeric = 0.0, scale: Numeric = 1.0, check: bool = True
    ):
        if check and np.any(scale < 0):
            raise NegativeParameterError(
                "In Normal(loc, scale), `scale` must be strictly positive!"
            )
        self.loc = loc
        self.scale = scale

    @cached_property
    def batch_shape(self) -> Shape:
        return np.broadcast(self.loc, self.scale).shape

    def logpdf(self, x: Numeric) -> Numeric:
        z = (x - self.loc) / self.scale
        return -np.square(z) / 2 - np.log(2 * np.pi) / 2 - np.log(self.scale)

    def logcdf(self, x: Numeric) -> Numeric:
        return log_ndtr((x - self.mean) / self.scale)

    def cdf(self, x: Numeric) -> Numeric:
        return ndtr((x - self.mean) / self.scale)

    def survival(self, x: Numeric) -> Numeric:
        return 1 - self.cdf(x)

    def _sample(self, size: Shape, rng: RNG) -> ndarray:
        return rng.normal(loc=self.loc, scale=self.scale, size=size)

    @cached_property
    def mean(self) -> Numeric:
        return np.broadcast_to(self.loc, self.batch_shape)

    @cached_property
    def std(self) -> Numeric:
        return np.broadcast_to(self.scale, self.batch_shape)

    @cached_property
    def var(self) -> Numeric:
        return np.square(self.std)

    @cached_property
    def mode(self) -> Numeric:
        return self.mean

    @cached_property
    def median(self) -> Numeric:
        return self.mean


class MvNormal(MultivariateReal):
    def __init__(
        self,
        mean: Optional[ndarray] = None,
        cov: Optional[ndarray] = None,
        **kwargs,
    ):
        match mean, cov:
            case (None, None):
                raise ValueError("mean and cov cannot both be None!")
            case (None, _):
                mean = np.zeros(cov.shape[-1])
            case (_, None):
                cov = np.eye(mean.shape[-1])

        super().__init__(**kwargs)

        self._mean = mean
        self._cov = cov

    @cached_property
    def mean(self) -> ndarray:
        return np.broadcast_to(self._mean, self.batch_plus_event_shape)

    @cached_property
    def _icov(self) -> ndarray:
        return np.linalg.inv(self._cov)

    @cached_property
    def cov(self) -> ndarray:
        return np.broadcast_to(
            self._cov, self.batch_plus_event_shape + self.event_shape
        )

    @cached_property
    def cov_inv(self) -> ndarray:
        return np.broadcast_to(
            self._icov, self.batch_plus_event_shape + self.event_shape
        )

    @cached_property
    def L(self) -> ndarray:
        return np.linalg.cholesky(self.cov)

    @cached_property
    def event_shape(self) -> Shape:
        return self.mean.shape[-1:]

    @cached_property
    def batch_shape(self) -> Shape:
        return self.batch_plus_event_shape[:-1]

    @cached_property
    def batch_plus_event_shape(self) -> Shape:
        return np.broadcast_shapes(self._mean.shape, self._cov.shape[:-1])

    @cached_property
    def log_det_cov(self) -> float | ndarray:
        _, ldc = np.linalg.slogdet(self.cov)
        return ldc

    @cached_property
    def var(self) -> ndarray:
        return np.diagonal(self.cov, axis1=-2, axis2=-1)

    def logpdf(self, x):
        d = x - self.mean

        # Compute quadratic form
        quad_form = np.einsum("...i, ...ij, ...j -> ...", d, self.cov_inv, d)  # type: ignore

        return -0.5 * (
            self.event_shape[0] * np.log(2 * np.pi)
            + self.log_det_cov
            + quad_form
        )

    def sample(
        self, sample_shape: Shape = (), rng: RNG = default_rng()
    ) -> ndarray:
        shape = sample_shape + self.batch_shape + self.event_shape
        standard_normals = rng.standard_normal(shape)
        b = np.einsum("...ij,...j->...i", self.L, standard_normals)
        samples = self.mean + b
        return samples


class Dirichlet(MultivariateContinuous):
    def __init__(self, concentration: ndarray, check: bool = True):
        if check and np.any(concentration < 0):
            raise NegativeParameterError(
                "In Dirichlet(concentration), `concentration` must be stricly positive!"
            )
        self.concentration = concentration

    @cached_property
    def concentration_sum(self):
        return self.concentration.sum(-1, keepdims=True)

    @cached_property
    def event_shape(self):
        return self.concentration.shape[-1]

    @cached_property
    def batch_plus_event_shape(self):
        return self.concentration.shape

    @cached_property
    def batch_shape(self):
        return self.batch_plus_event_shape[:-1]

    def logpdf(self, x: ndarray) -> float | ndarray:
        # TODO: Test.
        return (
            np.sum((self.concentration - 1) * np.log(x), -1)
            + gammaln(self.concentration.sum(-1))
            - gammaln(self.concentration).sum(-1)
        )

    def sample(
        self, sample_shape: Shape = (), rng: RNG = default_rng()
    ) -> ndarray:
        shape = sample_shape + self.batch_plus_event_shape
        alpha = rng.standard_gamma(shape)
        return alpha / alpha.sum(-1, keepdims=True)

    def to_real(self, x: ndarray) -> float | ndarray:
        # https://mc-stan.org/docs/2_19/reference-manual/simplex-transform-section.html
        raise NotImplementedError

    def to_native(self, z: ndarray) -> float | ndarray:
        # https://mc-stan.org/docs/2_19/reference-manual/simplex-transform-section.html
        raise NotImplementedError

    def logdetjac(self, z: ndarray) -> float | ndarray:
        # https://mc-stan.org/docs/2_19/reference-manual/simplex-transform-section.html
        raise NotImplementedError

    @cached_property
    def cov(self) -> ndarray:
        raise NotImplementedError

    @cached_property
    def mean(self) -> ndarray:
        return self.concentration / self.concentration_sum

    @cached_property
    def var(self) -> ndarray:
        m = self.mean
        return m * (1 - m) / (1 + self.concentration_sum)

    @cached_property
    def std(self) -> ndarray:
        return np.sqrt(self.var)
