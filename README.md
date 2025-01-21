# Arianna

[![CI Status][ci-status-img]](https://github.com/lanl/arianna-ppl/actions)
[![PyPI Version][pypi-version]](https://pypi.org/project/arianna-ppl/)
[![PyPI Downloads][pypi-downloads]](https://pypistats.org/packages/arianna-ppl)

A probabilistic programming language for python built on `numpy`.

## Installation
To install from PyPI (recommended):

**pip**
```
pip install arianna-ppl
```

**uv**
```
uv add arianna-ppl
```

To install the absolute latest version from `main`:

**pip**
```
pip install git+https://github.com/lanl/arianna-ppl.git
```

**uv**
```
uv add git+https://github.com/lanl/arianna-ppl.git
```


## Usage

**Model Specification (linear regression)**
```python
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

# Type annotation are, of course, optional. Provided only for clarity.
def linear_regression(
    ctx: Context,
    X: np.ndarray,
    y: Optional[np.ndarray]=None,
    bias: bool=True
) -> None:
    _, p = X.shape
    beta = ctx.rv("beta", Normal(np.zeros(p), 10))
    sigma = ctx.rv("sigma", Gamma(1, 1))
    mu = ctx.cached("mu", X @ beta)
    if bias:
        alpha = ctx.rv("alpha", Normal(0, 10))
        mu += alpha

    ctx.rv("y", Normal(mu, sigma), obs=y)
```

**Simulate data from Prior Predictive**
```python
nobs = 100
rng = np.random.default_rng(0)

# Generate random predictors (X).
X = rng.normal(0, 1, (nobs, 1))

# Simulate from prior predictive using Predictive.
sim_truth = Predictive.run(
    linear_regression,  # supplied model here.
    state=dict(sigma=0.7),
    rng=rng,
    X=X,
    # since y is None, the returned dictionary will contain y sampled from it's
    # predictive distributions.
    y=None,
    # Not return cached values, so the sim_truth will contain only parameters
    # and y.
    return_cached=False,  
)

# pop y so that sim_truth contains only model parameters.
y = sim_truth.pop("y")

# Now sim_truth is a dict containing ("beta", "sigma", "alpha").
```

**Affine invariant ensemble sampler**
```python
aies = AIES(
    linear_regression,  # model function.
    nwalkers=10,  # number of walkers.
    # Whether or not to transform parameters into unconstrained space.
    transform=True,  # Set to true when possible.
    # Random number generator for reproducibility.
    rng=default_rng(0),
    # Provide data.
    X=X, y=y,
)

# Does 3000 steps, with 10 walkers, after burning for 3000, and thins by 1. At
# the end, 3000 = 3000*10 samples will be aggregated from all 10 walkers. Then,
# by default, these samples are passed into an importance sampler to reweight
# the samples, yielding 3000 samples.
chain = aies.fit(nsteps=3000, burn=3000, thin=1)
```

`chain` is an object that contains posterior samples (states).
You can iterate over `chain`.

```python
for state in chain:
    print(state)  # state is a e.g., dict(alpha=1.3, beta=2.5, sigma=0.6, mu=some_long_array)
    break # just print the first one.
```

You can convert `chain` into a large dict with `bundle = chain.bundle`,
which is a `dict[str, ndarray]`.

You can also get the samples directly with `chain.samples`.

**Parallel Affine invariant ensemble sampler**
Works only in python 3.13t. But 3.13t does not yet work with `jupyter`.

```python
from concurrent.futures import ThreadPoolExecutor

paies = ParallelAIES(
    linear_regression,  # model function.
    ThreadPoolExecutor(4)  # use 4 cores.
    nwalkers=10,  # number of walkers.
    # Whether or not to transform parameters into unconstrained space.
    transform=True,  # Set to true when possible.
    # Random number generator for reproducibility.
    rng=default_rng(0),
    # Provide data.
    X=X, y=y,
)

# Same as non-parallel version, but will be faster in python 3.13t.
# Will be slightly slower than the non-parallel version in GIL enabled python
# builds, i.e. python 3.9, 3.10, 3.11, 3.12, 3.13.
chain = paies.fit(nsteps=3000, burn=3000, thin=1)
```

**Laplace Approximation**
```python
la = LaplaceApproximation(
    linear_regression,
    transform=True,
    rng=default_rng(0),
    X=X, y=y,
)

# The MAP estimate and inverse Hessian are computed via L-BFGS optimization.
# Those estimates are used to construct a MvNormal object. 3000 samples are
# drawn from that resulting MvNormal.
chain = la.fit(nsamples=3000)
```

**Posterior Predictive**
```python
rng = default_rng
xnew = np.linspace(-3, 3, 50)
Xnew = xnew.reshape(-1, 1)
ynew = Chain(
    Predictive.run(
        linear_regression, state=state, rng=rng, X=Xnew, y=None
    )
    for state in chain
).get("y")
```

See [demos](demos/).

## Threading
As of 8 Jan 2025, `jupyter` does not work with the threaded (no-gil) version of
python 3.13 (3.13t). You can install `arianna` with python 3.13 or python 3.13t
but you cannot install `jupyter` also. If you must use `jupyter`, use python
3.10, 3.11, 3.12, 3.13 (but not 3.13t).

## LANL Software Release Information
- O4856

[ci-status-img]: https://img.shields.io/github/actions/workflow/status/lanl/arianna-ppl/CI.yml?style=flat-square&label=CI
[pypi-version]: https://img.shields.io/pypi/v/arianna-ppl?style=flat-square&label=PyPI
[pypi-downloads]: https://img.shields.io/pypi/dm/arianna-ppl?style=flat-square&label=Downloads&color=blue

