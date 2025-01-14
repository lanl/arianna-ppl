from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from numpy import ndarray
from numpy.random import Generator
from numpy.testing import assert_allclose
from scipy import stats

import arianna.distributions as dist
from arianna.distributions.abstract import MultivariateContinuous


def make_stacked_diag_cov(
    batch_shape: tuple[int, ...], dim: int, rng: Generator
):
    cov = np.empty(batch_shape + (dim, dim))
    for indices in np.ndindex(batch_shape):
        cov[indices] = np.diag(np.abs(rng.normal(0, 3, dim)))
    return cov


class AbstractTestMulivariateContinuous(ABC):
    @cached_property
    @abstractmethod
    def d(self) -> MultivariateContinuous: ...

    @cached_property
    @abstractmethod
    def x(self) -> ndarray: ...

    @cached_property
    @abstractmethod
    def logpdf_truth(self) -> ndarray: ...

    @property
    def event_shape(self):
        return (2,)

    @property
    def batch_shape(self):
        return (3,)

    @property
    def sample_shape(self):
        return (7, 5)

    @cached_property
    def rng(self):
        return np.random.default_rng(1)

    def test_sample(self):
        assert (
            np.shape(self.d.sample(rng=self.rng))
            == self.batch_shape + self.event_shape
        )
        assert (
            np.shape(self.d.sample(self.sample_shape, rng=self.rng))
            == self.sample_shape + self.batch_shape + self.event_shape
        )

        samples = self.d.sample((100_000,), rng=self.rng)
        assert_allclose(samples.mean(0), self.d.mean, rtol=0.05)
        assert_allclose(samples.std(0), self.d.std, rtol=0.05)

    def test_logpdf(self):
        lpdf = self.d.logpdf(self.x)
        assert_allclose(self.logpdf_truth, lpdf, rtol=1e-6)

        pdf = self.d.pdf(self.x)
        assert_allclose(np.exp(lpdf), pdf, rtol=1e-6)


class TestMvNormal(AbstractTestMulivariateContinuous):
    @cached_property
    def d(self):
        return dist.MvNormal(
            mean=self.rng.normal(0, 1, (self.batch_shape + self.event_shape)),
            cov=make_stacked_diag_cov(
                self.batch_shape, self.event_shape[0], self.rng
            ),
        )

    @cached_property
    def x(self):
        return self.rng.normal(
            0.0, 1.0, (self.sample_shape + self.batch_shape + self.event_shape)
        )

    @cached_property
    def logpdf_truth(self):
        return stats.norm.logpdf(
            self.x,
            self.d.mean,
            np.sqrt(np.diagonal(self.d.cov, axis1=-2, axis2=-1)),
        ).sum(-1)
