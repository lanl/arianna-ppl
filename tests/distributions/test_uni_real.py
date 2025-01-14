from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from matplotlib.pylab import Generator
from numpy import ndarray
from numpy.testing import assert_allclose
from scipy import stats

import arianna.distributions as dist
from arianna.distributions.abstract import UnivariateContinuous


# @pytest.mark.filterwarnings("ignore")
class AbstractTestUnivariateContinuous(ABC):
    @cached_property
    @abstractmethod
    def d(self) -> UnivariateContinuous: ...

    @cached_property
    @abstractmethod
    def x(self) -> ndarray: ...

    @cached_property
    @abstractmethod
    def pdf_truth(self) -> ndarray: ...

    @cached_property
    @abstractmethod
    def cdf_truth(self) -> ndarray: ...

    @cached_property
    def rng(self) -> Generator:
        return np.random.default_rng(0)

    def test_sample(self):
        m = (2,)
        assert np.shape(self.d.sample()) == m

        n = (7, 5)
        assert np.shape(self.d.sample(n)) == n + m

        samples = self.d.sample((100_000,), rng=self.rng)
        assert np.allclose(samples.mean(0), self.d.mean, rtol=0.02)
        assert np.allclose(samples.std(0), self.d.std, rtol=0.02)

    def test_logpdf(self):
        pdf = self.d.pdf(self.x)
        assert_allclose(self.pdf_truth, pdf, rtol=1e-6)

        lpdf = self.d.logpdf(self.x)
        assert_allclose(np.exp(lpdf), pdf, rtol=1e-6)

    def test_logcdf(self):
        cdf = self.d.cdf(self.x)
        assert_allclose(self.cdf_truth, cdf, rtol=1e-6)

        lcdf = self.d.logcdf(self.x)
        assert_allclose(np.exp(lcdf), cdf, rtol=1e-6)

        survival = self.d.survival(self.x)
        assert_allclose(1 - cdf, survival, rtol=1e-6)


class TestUniform(AbstractTestUnivariateContinuous):
    @cached_property
    def d(self):
        return dist.Uniform(lower=np.array([0, -5]), upper=np.array([1, 11]))

    @cached_property
    def x(self):
        return np.array([[0.6, 6], [0, 11], [-1, 12]])

    @cached_property
    def pdf_truth(self):
        return np.where(
            (self.d.lower < self.x) & (self.x < self.d.upper),
            stats.uniform.pdf(self.x, loc=self.d.lower, scale=self.d.range),
            0,
        )

    @cached_property
    def cdf_truth(self):
        return stats.uniform.cdf(self.x, loc=self.d.lower, scale=self.d.range)

    def test_from_mean_shift(self):
        d = dist.Uniform.from_mean_shift(3, 1)
        assert d.lower == 2
        assert d.upper == 4

    def _test_logjacdet(self):
        # TODO.

        # Case 1:
        # Uniform(0 ,1)
        # Beta(2,4)

        # Can we 2:
        # Uniform(-2, 3)

        # Case 3:
        # X ~ Uniform(1, 3)
        # Y|X ~ Uniform(X, 5)
        # Compare against analytic marginal distribution for Y.

        # def case1(ctx: Context):
        #     alpha = ctx.rv("alpha", dist.Uniform(0, 1))
        #     beta = ctx.rv("beta", dist.Beta(2, 4))

        # def case2(ctx: Context):
        #     alpha = ctx.rv("alpha", dist.Uniform(-2, 3))

        # def case3(ctx: Context):
        #     alpha = ctx.rv("alpha", dist.Uniform(1, 3))
        #     beta = ctx.rv("beta", dist.Uniform(alpha, 5))

        # cases = [case1, case2, case3]
        # result = {}
        # for case in cases:
        #     for transform in (True, False):
        #         emcee = Emcee(case, transform=transform)
        #         result[case.__name__, transform] = emcee.fit(1000, burn=1000,
        #         thin=10)

        # for (case_name, transform), value in result.items():
        #     sns.pairplot(
        #         pd.DataFrame(value.bundle),
        #     )
        #     plt.suptitle(f"{case_name}, {transform}")
        #     plt.show()
        pass


class TestBeta(AbstractTestUnivariateContinuous):
    @cached_property
    def d(self):
        return dist.Beta(np.array([1, 2]), np.array([1, 3]))

    @cached_property
    def x(self):
        return np.array([[0.6, 0.6], [-10, 10], [0, 1]])

    @cached_property
    def pdf_truth(self):
        return np.where(
            (0 < self.x) & (self.x < 1),
            stats.beta.pdf(self.x, a=self.d.a, b=self.d.b),
            0,
        )

    @cached_property
    def cdf_truth(self):
        return stats.beta.cdf(self.x, a=self.d.a, b=self.d.b)


class TestGamma(AbstractTestUnivariateContinuous):
    @cached_property
    def d(self):
        return dist.Gamma(shape=np.array([3, 7]), scale=np.array([2, 4]))

    @cached_property
    def x(self):
        return np.array([[0.6, 6], [0, 11], [-1, 12]])

    @cached_property
    def pdf_truth(self):
        return stats.gamma.pdf(self.x, a=self.d.shape, scale=self.d.scale)

    @cached_property
    def cdf_truth(self):
        return stats.gamma.cdf(self.x, a=self.d.shape, scale=self.d.scale)

    def test_from_mean_std(self):
        d = dist.Gamma(3, 5)
        assert d.mean == 15
        assert d.var == 3 * 5**2

        m = self.rng.uniform(0, 10, 100)
        s = self.rng.uniform(0, 10, 100)
        d = dist.Gamma.from_mean_std(mean=m, std=s)
        assert_allclose(d.mean, m)
        assert_allclose(d.std, s)


class TestInverseGamma(AbstractTestUnivariateContinuous):
    @cached_property
    def d(self):
        return dist.InverseGamma(shape=np.array([3, 4]), scale=np.array([2, 3]))

    @cached_property
    def x(self):
        return np.array([[0.6, 6], [0, 5], [-1, 3]])

    @cached_property
    def pdf_truth(self):
        return stats.invgamma.pdf(self.x, a=self.d.shape, scale=self.d.scale)

    @cached_property
    def cdf_truth(self):
        return stats.invgamma(self.d.shape, scale=self.d.scale).cdf(self.x)

    def test_from_mean_std(self):
        m = self.rng.uniform(0, 5, 10)
        s = self.rng.uniform(0, 5, 10)
        d = dist.InverseGamma.from_mean_std(mean=m, std=s)
        assert_allclose(d.mean, m)
        assert_allclose(d.std, s)


class TestLogNormal(AbstractTestUnivariateContinuous):
    def test_sample(self):
        pass

    @cached_property
    def d(self):
        return dist.LogNormal(mu=np.array([3, 4]), sigma=np.array([2, 3]))

    @cached_property
    def x(self):
        return np.array([[0.6, 6], [0, 5], [-1, 3]])

    @cached_property
    def pdf_truth(self):
        return stats.lognorm.pdf(
            self.x, s=self.d.sigma, scale=np.exp(self.d.mu)
        )

    @cached_property
    def cdf_truth(self):
        return stats.lognorm(s=self.d.sigma, scale=np.exp(self.d.mu)).cdf(
            self.x
        )

    def test_from_mean_std(self):
        m = self.rng.uniform(0, 5, 10)
        s = self.rng.uniform(0, 5, 10)
        d = dist.LogNormal.from_mean_std(mean=m, std=s)
        assert_allclose(d.mean, m)
        assert_allclose(d.std, s)


class TestNormal(AbstractTestUnivariateContinuous):
    @cached_property
    def d(self):
        return dist.Normal(loc=np.array([-1, 2]), scale=np.array([2, 4]))

    @cached_property
    def x(self):
        return np.array([[0.6, 6], [0, 1], [-1, 5]])

    @cached_property
    def pdf_truth(self):
        return stats.norm.pdf(self.x, loc=self.d.loc, scale=self.d.scale)

    @cached_property
    def cdf_truth(self):
        return stats.norm.cdf(self.x, loc=self.d.loc, scale=self.d.scale)
