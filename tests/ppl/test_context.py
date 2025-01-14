import numpy as np
from numpy import array, ndarray
from numpy.random import default_rng

from arianna.distributions import Normal
from arianna.ppl.context import Context, LogprobAndTrace, Predictive


class TestContext:
    def test_gaussian_model(self):
        def model(ctx: Context, y: ndarray):
            mu = ctx.rv("mu", Normal(0, 10))
            log_sigma = ctx.rv("log_sigma", Normal(array(0), array(3)))
            sigma = ctx.cached("sigma", np.exp(log_sigma))
            ctx.rv("y", Normal(mu, sigma), obs=y)

        np.random.seed(0)
        state = {"mu": 0.0, "log_sigma": 0.0}
        y = np.random.normal(3, 2, 50)

        state = Predictive.run(model, rng=default_rng(0), y=y)
        lpdf, state_and_cache = LogprobAndTrace.run(model, state=state, y=y)
        assert lpdf > -np.inf
        assert "sigma" in state_and_cache

        for name in ("mu", "log_sigma", "sigma"):
            assert name in state
