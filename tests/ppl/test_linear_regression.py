from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from numpy.random import default_rng

from arianna.distributions import Gamma, Normal
from arianna.ppl.context import Context, Predictive
from arianna.ppl.inference import (
    AIES,
    AffineInvariantMCMC,
    Chain,
    LaplaceApproximation,
    ParallelAIES,
    RandomWalkMetropolis,
)

print("numpy version:", np.__version__)


def linear_regression(
    ctx: Context, X: np.ndarray, y: Optional[np.ndarray], bias=True
):
    _, p = X.shape
    beta = ctx.rv("beta", Normal(np.zeros(p), 10))
    sigma = ctx.rv("sigma", Gamma(1, 1))
    mu = ctx.cached("mu", X @ beta)
    if bias:
        alpha = ctx.rv("alpha", Normal(0, 10))
        mu += alpha

    ctx.rv("y", Normal(mu, sigma), obs=y)


def test_linear_regression():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (100, 1))
    sim_truth = Predictive.run(
        linear_regression,
        state=dict(sigma=0.7),
        rng=rng,
        X=X,
        y=None,
        return_cached=False,
    )
    y = sim_truth.pop("y")

    for transform in (True, False):
        proposal = {
            name: (
                lambda value, rng, mcmc_iteration: rng.normal(
                    value, np.clip(10 * np.exp(-mcmc_iteration / 100), 0.1, 5)
                )
            )
            for name in sim_truth
        }
        rwm = RandomWalkMetropolis(
            linear_regression,
            proposal=proposal,
            rng=default_rng(0),
            X=X,
            y=y,
            transform=transform,
        )
        aies = AIES(
            linear_regression,
            rng=default_rng(0),
            X=X,
            y=y,
            transform=transform,
        )
        paies = ParallelAIES(
            linear_regression,
            ThreadPoolExecutor(4),
            rng=default_rng(0),
            X=X,
            y=y,
            transform=transform,
        )
        laplace = LaplaceApproximation(
            linear_regression,
            rng=default_rng(2),
            X=X,
            y=y,
            transform=transform,
        )

        samplers = dict(laplace=laplace, rwm=rwm, aies=aies, paies=paies)

        for name, sampler in samplers.items():
            print(f"Transformed: {transform}, Sampler: {name}")
            n, burn = 3000, 3000
            match sampler:
                case AffineInvariantMCMC():
                    samples = sampler.fit(n, burn=burn, thin=1)
                case LaplaceApproximation():
                    samples = sampler.fit(n)
                case _:
                    thin = aies.nwalkers
                    samples = sampler.fit(n, burn=burn * thin, thin=thin)

            match sampler:
                case AffineInvariantMCMC():
                    print(f"{name} acceptance rates:", sampler.accept_rate)
                    # Authors say that acceptance rates in (0.2, 0.5) is best.
                    # Increase `a` to decrease acceptance frequency (bigger
                    # steps).
                    # Decrease `a` to increase acceptance frequency (smaller
                    # steps).
                    np.testing.assert_array_less(sampler.accept_rate, 1)
                    np.testing.assert_array_less(0, sampler.accept_rate)

            xnew = np.linspace(-3, 3, 50)
            Xnew = xnew.reshape(-1, 1)
            _ = Chain(
                Predictive.run(
                    linear_regression, state=c, rng=rng, X=Xnew, y=None
                )
                for c in samples
            ).get("y")  # ynew

            for name, value in samples.bundle.items():
                if name in sim_truth:
                    value = value.squeeze()

                    match name, transform, sampler:
                        case "sigma", False, LaplaceApproximation():
                            rtol = 0.1
                        case _:
                            rtol = 0.05

                    # Test that the posterior mean is near the truth.
                    np.testing.assert_allclose(
                        np.squeeze(sim_truth[name]),
                        samples.get(name).mean(),
                        rtol=rtol,
                    )

                    # Test that the 95% credible interval contains the truth.
                    assert np.all(np.quantile(value, 0.025) < sim_truth[name])
                    assert np.all(np.quantile(value, 0.975) > sim_truth[name])
