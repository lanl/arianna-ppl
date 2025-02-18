from abc import ABC, abstractmethod
from concurrent.futures import Executor
from copy import deepcopy
from functools import cached_property
from typing import (
    Any,
    Callable,
    Concatenate,
    Iterable,
    Iterator,
    Optional,
    ParamSpec,
)

import numpy as np
from numpy import ndarray
from numpy.random import Generator as RNG
from numpy.random import default_rng
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from tqdm import tqdm, trange

from arianna.distributions import MvNormal
from arianna.ppl.context import (
    Context,
    LogprobAndLogjacobianAndTrace,
    LogprobAndPriorSample,
    LogprobAndTrace,
    Predictive,
    State,
    TransformedLogprobAndTrace,
    TransformedPredictive,
)
from arianna.ppl.shaper import Shaper

P = ParamSpec("P")
Model = Callable[Concatenate[Context, P], None]
Logprob = Callable[[State], tuple[float, State]]


class Chain:
    """Chain MCMC samples.

    Parameters
    ----------
    states : Iterable[State]
        MCMC states.

    Attributes
    ----------
    chain : list[State]
        MCMC states in list format.
    """

    def __init__(self, states: Iterable[State]):
        self.states = list(states)
        self.names = list(self.states[0].keys())

    def __iter__(self) -> Iterator[State]:
        """Iterate over states.

        Yields
        ------
        State
            MCMC state within chain.
        """
        for state in self.states:
            yield state

    def __len__(self) -> int:
        """Return the number of states."""
        return len(self.states)

    def get(self, name: str) -> ndarray:
        """Get all MCMC samples for one variable or cached value by name.

        Parameters
        ----------
        name : str
            Name of model parameter or cached value.

        Returns
        -------
        ndarray
            MCMC samples for the variable or cached value named.
        """
        return np.stack([c[name] for c in self.states])

    @cached_property
    def bundle(self) -> dict[str, ndarray]:
        """Bundle MCMC values into a dictionary.

        Returns
        -------
        dict[str, ndarray]
            Dictionary bundle of MCMC samples.
        """
        return {name: self.get(name) for name in self.names}

    def subset(self, burn: int = 0, thin: int = 1):
        """Return subset of the states.

        Parameters
        ----------
        burn : int, optional
            Number of initial samples to discard, by default 0.
        thin : int, optional
            Take only every `thin`-th sample, by default 1.

        Returns
        -------
        Chain
            A whole new chain, with the first `burn` removed, and taking only
            every `thin`-th sample.
        """
        return Chain(self.states[burn::thin])


class InferenceEngine(ABC):
    """Abstract inference engine class."""

    rng: RNG

    @abstractmethod
    def fit(self):
        """Fit model."""
        pass


class MCMC(InferenceEngine):
    """Abstract class for MCMC."""

    model: Model
    model_data: dict[str, Any]
    nsamples: int
    burn: int
    thin: int
    mcmc_iteration: int
    transform: bool
    logprob_history: list[float]

    @abstractmethod
    def _fit(self, *args, **kwargs) -> Iterator[State]:
        pass

    @abstractmethod
    def step(self):
        """Update model state in one MCMC iteration."""
        pass

    def fit(self, *args, **kwargs) -> Chain:
        """Run MCMC.

        Returns
        -------
        Chain
            Chain of MCMC samples.
        """
        return Chain(self._fit(*args, **kwargs))

    def logprob(self, state: State) -> tuple[float, State]:
        """Compute log density.

        Parameters
        ----------
        state : State
            Dictionary containing random variables to model.

        Returns
        -------
        tuple[float, State]
            (Log density (float), native state and cached values (dict))
        """
        if self.transform:
            # state is real.
            # Returns logprob + log_det_jacobian, native_state.
            return TransformedLogprobAndTrace.run(
                self.model, state, **self.model_data
            )
        else:
            # state is native.
            # Returns logprob, state (which is already native).
            return LogprobAndTrace.run(self.model, state, **self.model_data)


class SingleWalkerMCMC(MCMC):
    """Markov Chain Monte Carlo."""

    init_state: State
    mcmc_state: State
    transform: bool

    @abstractmethod
    def step(self) -> tuple[float, State]:
        """Update mcmc_state and return logprob and native_state_and_cache.

        Returns
        -------
        float, State
            Logprob and native state and cache dictionary.
        """
        pass

    def _fit(
        self, nsamples: int, burn: int = 0, thin: int = 1
    ) -> Iterator[State]:
        self.nsamples = nsamples
        self.burn = burn
        self.thin = thin
        self.mcmc_state = deepcopy(self.init_state)
        self.logprob_history = []

        for i in trange(nsamples * thin + burn):
            self.mcmc_iteration = i
            # NOTE: mcmc_state may not be the returned state, but the state
            # that is used in the MCMC (e.g., for computational efficiency).
            # trace is the state in its native space appended with any cached
            # values.
            logprob, trace = self.step()
            if i >= burn and (i + 1) % thin == 0:
                self.logprob_history.append(logprob)
                yield trace


class RandomWalkMetropolis(SingleWalkerMCMC):
    """Random walk Metropolis.

    Parameters
    ----------
    model: Model
        model function.
    init_state: Optional[State]
        Initial state for MCMC. If `transform=True` then `init_state` should
        contain values in the real space; if `transform=False`, then
        `init_state` should contain values in the native space. If not
        provided, `init_state` is sampled from the prior predictive.
        Defaults to None.
    proposal: Optional[dict[str, Any]]
        Dictionary containing proposal functions, dependent on the current
        value. Defaults to None.
    transform: bool
        Whether or not to sample parameters into the real space. If False,
        samples parameters in the native space. Regardless, returned samples
        are in the native space and will include cached values. Defaults to
        True.
    rng: Optional[RNG]
        Numpy random number generator. Defaults to None.

    Attributes
    ----------
    model: Model
        See Parameters.
    init_state: State
        If the constructor received None, `init_state` will be an empty
        dictionary.
    proposal: dict[str, Any]
        If None is received in the constructor, an empty dictionary is first
        created. In addition, any model parameters unnamed in the constructor
        will have a value of
        `lambda value, rng, mcmc_iteration: rng.normal(value, 0.1)`.
        Thus, if you supplied in the constructor
        `dict(mu=lambda value, rng, mcmc_iteration: rng.normal(value, 1))`
        and your model is
        ```python
        def model(ctx, y=None):
            mu = ctx.rv("mu", Normal(0, 10))
            sigma = ctx.rv("sigma", Gamma(1, 1))
            ctx.rv("y", Normal(mu, sigma), obs=y)
        ```
        then the value for sigma will be
        `lambda value, rng, _: rng.normal(value, 0.1)`.
    transform: bool
        See Parameters.
    rng: RNG
        If `None` was supplied in the constructor, then rng will be set to
        `np.random.default_rng()`.
    """

    mcmc_state: State
    init_state: State

    def __init__(
        self,
        model: Model,
        init_state: Optional[State] = None,
        proposal: Optional[dict[str, Any]] = None,
        transform: bool = True,
        rng: Optional[RNG] = None,
        **model_data,
    ):
        self.model = model
        self.model_data = model_data
        self.transform = transform
        self.rng = rng or default_rng()

        match init_state, transform:
            case None, True:
                self.init_state = TransformedPredictive.run(
                    model, rng=rng, return_cached=False, **model_data
                )
            case None, False:
                self.init_state = Predictive.run(
                    model, rng=rng, return_cached=False, **model_data
                )
            case _:  # init_state is provided.
                self.init_state = init_state

        self.proposal = proposal or {}
        for name in self.init_state:  # should not have cached values.
            self.proposal.setdefault(
                name, lambda value, rng, _: rng.normal(value, 0.1)
            )

    def step(self) -> tuple[float, State]:
        """Update mcmc_state and return native state and cached values.

        Returns
        -------
        State
            Native state and cached values.
        """
        proposed_state = {
            name: propose(self.mcmc_state[name], self.rng, self.mcmc_iteration)
            for name, propose in self.proposal.items()
        }
        # NOTE: A trace contains the state (i.e., result of rv) in the native
        # space AND cached values (i.e., result of cached).
        logprob_proposed, proposed_trace = self.logprob(proposed_state)
        logprob_current, current_trace = self.logprob(self.mcmc_state)
        if logprob_proposed - logprob_current > np.log(self.rng.uniform()):
            self.mcmc_state = proposed_state
            return logprob_proposed, proposed_trace
        else:
            return logprob_current, current_trace


class AffineInvariantMCMC(MCMC):
    """Affine Invariant MCMC."""

    nsteps: int
    init_state: list[State]
    mcmc_state: list[State]
    accept_rate: ndarray
    accept: list[int]
    nwalkers: int
    rng: RNG
    a: float

    @cached_property
    def dim(self) -> int:
        """Number of model parameters."""
        return int(
            sum(
                np.prod(np.shape(value))
                for value in self.init_state[0].values()
            )
        )

    def logprob(self, state: State) -> tuple[float, float, State]:
        """Compute log density.

        Parameters
        ----------
        state : State
            Dictionary containing random variables to model.

        Returns
        -------
        tuple[float, float, State]
            (
                Log density in native space,
                Log determinant of jacobian,
                native state and cached values (dict)
            )
        """
        if self.transform:
            # state is real.
            # Returns logprob, log_det_jacobian, native_state.
            return LogprobAndLogjacobianAndTrace.run(
                self.model, state, **self.model_data
            )
        else:
            # state is native.
            # Returns logprob, logprob, state (which is already native).
            lp, trace = LogprobAndTrace.run(
                self.model, state, **self.model_data
            )
            return lp, 0, trace

    @abstractmethod
    def step(self) -> tuple[list[float], list[State]]:
        """Update mcmc_state and return list of native_state_and_cache.

        Returns
        -------
        list[State]
            List of native state and cache dictionary.
        """
        pass

    def _update_walker(self, i: int) -> tuple[float, State]:
        this_walker = self.mcmc_state[i]
        z = self._draw_z(i)
        other_walker = self._draw_walker(i)

        candidate = {
            name: value + (this_walker[name] - value) * z
            for name, value in other_walker.items()
        }

        cand_logprob, cand_ldj, cand_trace = self.logprob(candidate)
        this_logprob, this_ldj, this_trace = self.logprob(this_walker)
        log_accept_prob = cand_logprob + cand_ldj - this_logprob - this_ldj
        log_accept_prob += (self.dim - 1) * np.log(z)
        if log_accept_prob > np.log(self._draw_u(i)):
            if self.mcmc_iteration >= self.burn:
                self.accept[i] += 1
            for key, value in candidate.items():
                this_walker[key] = value
            trace = cand_trace
            lp = cand_logprob
        else:
            trace = this_trace
            lp = this_logprob

        return lp, trace

    def fit(
        self, *args, rebalanced_samples: Optional[int] = None, **kwargs
    ) -> Chain:
        """Fit model with AIES."""
        chain = Chain(self._fit(*args, **kwargs))

        if rebalanced_samples is None:
            rebalanced_samples = self.nsteps

        if rebalanced_samples > 0:
            # Reweight with importance sampling.
            weights = softmax(self.logprob_history)
            index = self.rng.choice(
                len(weights), rebalanced_samples, replace=True, p=weights
            )
            self.resampled_logprob_history = np.array(
                [self.logprob_history[i] for i in index]
            )
            chain = Chain(chain.states[i] for i in index)

        return chain

    def _fit(
        self, nsteps: int, burn: int = 0, thin: int = 1
    ) -> Iterator[State]:
        self.nsteps = nsteps
        self.nsamples = nsteps * self.nwalkers
        self.burn = burn
        self.thin = thin
        self.mcmc_state = deepcopy(self.init_state)
        self.logprob_history = []

        for i in trange(nsteps * thin + burn):
            self.mcmc_iteration = i
            # NOTE: mcmc_state may not be the returned state, but the state
            # that is used in the MCMC (e.g., for computational efficiency).
            # trace is the state in its native space appended with any cached
            # values.
            logprob, trace = self.step()
            if i >= burn and (i + 1) % thin == 0:
                self.logprob_history.extend(logprob)
                yield from trace

        self.accept_rate = np.array(self.accept) / (self.nsteps * self.thin)

    @cached_property
    def _root_a(self) -> float:
        return np.sqrt(self.a)

    @cached_property
    def _invroot_a(self) -> float:
        return 1 / self._root_a

    @abstractmethod
    def _draw_walker(self, i: int) -> State: ...

    @abstractmethod
    def _draw_u(self, i: int) -> float: ...

    def _compute_z_given_u(self, u: float) -> float:
        return (u * (self._root_a - self._invroot_a) + self._invroot_a) ** 2

    def _draw_z(self, i: int) -> float:
        u = self._draw_u(i)
        return self._compute_z_given_u(u)


# https://arxiv.org/abs/1202.3665
class AIES(AffineInvariantMCMC):
    """Sequential Affine Invariant Ensemble Sampler.

    This sampler is good for target distributions that are not multimodal and
    separated by large low density regions. You should use as many walkers as
    you can afford. Whereas this sampler employs walkers that are sequeutnailly
    updated.  there is a parallel analog that updates walkers in parallel.

    Parameters
    ----------
    model : Model
        A model function of the form `def model(ctx: Context, **data)`.
    num_walkers : int, optional
        Number of walkers. Defaults to 10.
    transform : bool, optional
        Whether or not to transform parameters into the real space, by default
        True.
    rng : RNG, optional
        Random number generator, by default default_rng()
    a : float, optional
        Tuning parameter that is set, by default, to 2.0, which is good for many
        cases.
    temperature_fn : Optional[Callable[[int], float]], optional
        A temperature function for annealing, by default None.

    References
    ----------
    - [emcee: The MCMC Hammer](https://arxiv.org/pdf/1202.3665)
    - [Ensemble Samplers with Affine Invariance](https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-s.pdf)
    """

    @staticmethod
    def default_temperature_fn(iter: int) -> float:
        """Return 1."""
        return 1.0

    def __init__(
        self,
        model: Model,
        nwalkers: int = 10,
        transform: bool = True,
        rng: RNG = default_rng(),
        a: float = 2.0,
        temperature_fn: Optional[Callable[[int], float]] = None,
        init_state: Optional[list[State]] = None,
        **model_data,
    ):
        self.model: Model = model
        self.nwalkers: int = nwalkers
        self.transform: bool = transform
        self.rng = rng
        self.accept = [0] * nwalkers
        if a <= 1:
            raise ValueError("Tuning parameter `a` must be larger than 1.")

        self.a: float = a
        self.model_data = model_data
        self.temperature_fn: Callable[[int], float] = (
            temperature_fn or self.default_temperature_fn
        )
        predictive = TransformedPredictive if transform else Predictive
        if init_state is None:
            init_state = [
                predictive.run(
                    model, rng=rng, return_cached=False, **model_data
                )
                for _ in range(self.nwalkers)
            ]
        self.init_state = init_state

    def step(self) -> tuple[list[float], list[State]]:
        """Update mcmc_state and return list of native_state_and_cache.

        Returns
        -------
        list[float], list[State]
            List of logprobs and list of native state and cache dictionary.
        """
        trace = []
        lp = []
        for i, _ in enumerate(self.mcmc_state):
            lp_i, trace_i = self._update_walker(i)
            lp.append(lp_i)
            trace.append(trace_i)

        return lp, trace

    def _draw_u(self, _: int) -> float:
        return self.rng.uniform()

    def _draw_walker(self, i: int) -> State:
        # Draw anything but the current walker (i).
        if (j := self.rng.integers(self.nwalkers)) == i:
            return self._draw_walker(i)
        else:
            return self.mcmc_state[j]


class ParallelAIES(AffineInvariantMCMC):
    """Parallel Affine Invariant MCMC (or Parallel AIES).

    References
    ----------
    - [emcee: The MCMC Hammer](https://arxiv.org/pdf/1202.3665)
    - [Ensemble Samplers with Affine Invariance](https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-s.pdf)
    """

    def __init__(
        self,
        model: Model,
        executor: Executor,
        nwalkers: int = 10,
        transform: bool = True,
        rng: RNG = default_rng(),
        a: float = 2.0,
        init_state: Optional[list[State]] = None,
        **model_data,
    ):
        if nwalkers < 4 or nwalkers % 2 == 1:
            raise ValueError(
                "num_walkers needs to be an even integer greater than 3, "
                f"but got {nwalkers}!"
            )

        self.executor = executor
        self.model: Model = model
        self.nwalkers: int = nwalkers
        self.transform: bool = transform
        self.rng = rng
        self.rngs = self.rng.spawn(self.nwalkers)
        self.accept = [0] * nwalkers
        if a <= 1:
            raise ValueError("Tuning parameter `a` must be larger than 1.")

        self.a: float = a
        self.model_data = model_data
        predictive = TransformedPredictive if transform else Predictive
        if init_state is None:
            init_state = [
                predictive.run(
                    model, rng=self.rng, return_cached=False, **model_data
                )
                for _ in range(self.nwalkers)
            ]
        self.init_state = init_state

    def _draw_u(self, i: int) -> float:
        return self.rngs[i].uniform()

    def step(self) -> tuple[list[float], list[State]]:
        """Update mcmc_state and return list of native_state_and_cache.

        Returns
        -------
        list[float], list[State]
            Tuple in which the first element is a list of logprobs, and the
            second element is a list of traces (i.e., native state and cache
            dictionary).
        """
        mid = self.nwalkers // 2
        out_first_half = list(
            self.executor.map(self._update_walker, range(mid))
        )
        out_second_half = list(
            self.executor.map(self._update_walker, range(mid, self.nwalkers))
        )
        logprob_first_half, trace_first_half = zip(*out_first_half)
        logprob_second_half, trace_second_half = zip(*out_second_half)

        logprob = logprob_first_half + logprob_second_half
        trace = trace_first_half + trace_second_half
        return logprob, trace

    def _draw_walker(self, i: int) -> State:
        other_walkers = self._get_other_walkers(i)
        j = self.rngs[i].integers(len(other_walkers))
        return other_walkers[j]

    def _get_other_walkers(self, i: int) -> list[State]:
        mid = self.nwalkers // 2
        if i < mid:
            return self.mcmc_state[mid:]
        else:
            return self.mcmc_state[:mid]


class ImportanceSampling(InferenceEngine):
    """Importance Sampling."""

    particles: list[State]

    def __init__(
        self,
        model: Model,
        rng: Optional[RNG] = None,
        particles: Optional[list[State]] = None,
        nparticles: Optional[int] = None,
        temperature: float = 1.0,
        **model_data,
    ):
        self.model = model
        self.model_data = model_data
        self.temperature = temperature
        self.rng = rng or default_rng()
        match nparticles, particles:
            case None, None:
                raise ValueError(
                    "nparticles and particles cannot both be None!"
                )
            case _, None:
                self.nparticles = nparticles
                logprobs_and_samples = [
                    LogprobAndPriorSample.run(
                        model=self.model, rng=self.rng, **self.model_data
                    )
                    for _ in trange(self.nparticles)
                ]
                self.logprobs, self.particles = zip(*logprobs_and_samples)
            case None, _:
                self.particles = particles
                self.nparticles = len(particles)
                self.logprobs = [
                    LogprobAndTrace.run(
                        model=self.model, state=particle, **self.model_data
                    )[0]
                    for particle in tqdm(self.particles)
                ]
            case _:
                raise ValueError(
                    "nparticles and particles cannot both be specified!"
                )

        self.log_weights = log_softmax(self.logprobs)
        self.weights = softmax(self.logprobs)
        # self.ess = ess_kish(self.weights)

    def fit(self, nsamples: int) -> Chain:
        """Sample."""
        indices = self.rng.choice(self.nparticles, nsamples, p=self.weights)
        return Chain(self.particles[i] for i in indices)


class LaplaceApproximation(InferenceEngine):
    """Laplace Approximation of Posterior."""

    rng: RNG

    def __init__(
        self,
        model: Model,
        transform: bool = True,
        rng: Optional[RNG] = None,
        init_state: Optional[State] = None,
        transformed_init_state: Optional[State] = None,
        **model_data,
    ):
        self.model = model
        self.model_data = model_data
        self.rng = rng or default_rng()
        self.transform = transform

        if self.transform:
            self.init_state = (
                transformed_init_state
                or TransformedPredictive.run(
                    model, rng=rng, return_cached=False, **self.model_data
                )
            )

        else:
            self.init_state = init_state or Predictive.run(
                model, rng=rng, return_cached=False, **self.model_data
            )

        self.shaper = Shaper.from_state(self.init_state)
        self.init_vec_state = self.shaper.vec(self.init_state)

    def logprob(self, vec_state: np.ndarray) -> float:
        """Compute log density.

        Parameters
        ----------
        state : State
            Dictionary containing random variables to model.

        Returns
        -------
        tuple[float, State]
            (Log density (float), native state and cached values (dict))
        """
        state = self.shaper.unvec(vec_state)
        if self.transform:
            # state is real.
            # Returns logprob + log_det_jacobian, native_state.
            return TransformedLogprobAndTrace.run(
                self.model, state, **self.model_data
            )[0]
        else:
            # state is native.
            # Returns logprob, state (which is already native).
            return LogprobAndTrace.run(self.model, state, **self.model_data)[0]

    def _negative_logprob(self, vec_state) -> float:
        return -self.logprob(vec_state)

    def fit(self, nsamples: int, **minimize_kwargs):
        """Fit model with laplace approx."""
        self.result = minimize(
            self._negative_logprob, x0=self.init_vec_state, **minimize_kwargs
        )
        mean = self.result.x
        cov = self.result.hess_inv
        samples = MvNormal(mean, cov).sample((nsamples,), rng=self.rng)

        # Return native state and cache.
        if self.transform:
            return Chain(
                TransformedLogprobAndTrace.run(
                    self.model,
                    self.shaper.unvec(vec_state),
                    **self.model_data,
                )[1]
                for vec_state in samples
            )
        else:
            return Chain(
                Predictive.run(
                    self.model,
                    self.shaper.unvec(vec_state),
                    return_cached=True,
                    **self.model_data,
                )
                for vec_state in samples
            )


class BayesianOptimization(InferenceEngine):
    """Bayesian Optimization."""

    ...


class AdaptiveRandomWalkMetropolis(SingleWalkerMCMC):
    """Adaptive Random Walk Metropolis.

    Resources
    ---------
    - https://probability.ca/jeff/ftpdir/adaptex.pdf
    """

    ...
