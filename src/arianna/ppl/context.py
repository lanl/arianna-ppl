from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol

import numpy as np
from numpy import ndarray
from numpy.random import Generator as RNG
from numpy.random import default_rng

from arianna.types import NegativeInfinityError, Numeric, Shape, State


class BasicDistribution(Protocol):
    def logpdf(self, x: Numeric) -> Numeric: ...
    def sample(
        self, sample_shape: Shape = (), rng: RNG = default_rng()
    ) -> ndarray: ...


class TransformableDistribution(BasicDistribution):
    def logdetjac(self, z: Numeric) -> Numeric: ...

    def logpdf_plus_logdetjac(self, z: Numeric) -> Numeric: ...

    def to_real(self, x: Numeric) -> Numeric:
        return np.log(x)

    def to_native(self, z: Numeric) -> Numeric:
        return np.exp(z)

    def logpdf(self, x: Numeric) -> Numeric: ...


# NOTE: Ideally, the name of a child Context class should be what its `run`
# method returns.  For example, the LogprobAndTrace Context's `run` method
# returns the model log probability and trace. (Note that a trace is the state
# in the native space and also includes cached values.)
class Context(ABC):
    result: Any
    state: State

    @classmethod
    @abstractmethod
    def run(cls): ...

    @abstractmethod
    def rv(
        self, name: str, dist: BasicDistribution, obs: Optional[Numeric] = None
    ) -> Numeric: ...

    @abstractmethod
    def cached(self, name: str, value: Numeric) -> Numeric: ...

    def __call__(self):
        return self.result


class LogprobAndPriorSample(Context):
    @classmethod
    def run(
        cls, model, rng: Optional[RNG] = None, **data
    ) -> tuple[float, State]:
        """Get (logprob, trace)."""
        ctx = cls(rng=rng)
        model(ctx, **data)
        return ctx.result

    def __init__(self, rng: Optional[RNG] = None):
        self.result = [0.0, {}]
        self.rng = rng or default_rng()

    def rv(
        self, name: str, dist: BasicDistribution, obs: Optional[Numeric] = None
    ):
        if obs is None:
            value = dist.sample()
            self.result[1][name] = value
        else:
            value = obs

        self.result[0] += np.sum(dist.logpdf(value))

        return value

    def cached(self, name: str, value: Numeric) -> Numeric:
        self.result[1][name] = value
        return value


class LogprobAndTrace(Context):
    @classmethod
    def run(cls, model, state: State, **data) -> tuple[float, State]:
        """TODO.

        Returns (logprob, trace). A trace is the state in the native space and
        the cached values.

        Parameters
        ----------
        model : Any
            _description_

        state : State
            _description_

        Returns
        -------
        tuple[float, State]
            _description_
        """
        ctx = cls(state)

        try:
            # Accumulate logprob.
            model(ctx, **data)
        except NegativeInfinityError:
            # If -inf anywhere during the accumulation, just end early and
            # return -inf and an empty trace.
            return -np.inf, {}

        return ctx.result

    def __init__(self, state: State):
        self.state = state
        self.result = [0.0, {}]

    def rv(
        self, name: str, dist: BasicDistribution, obs: Optional[Numeric] = None
    ):
        if obs is None:
            value = self.state[name]
            self.result[1][name] = value
        else:
            value = obs

        self.result[0] += np.sum(dist.logpdf(value))
        if self.result[0] == -np.inf:
            raise NegativeInfinityError("Negative infinity in Logprob.")

        return value

    def cached(self, name: str, value: Numeric) -> Numeric:
        self.result[1][name] = value
        return value


class Predictive(Context):
    @classmethod
    def run(
        cls,
        model,
        state: Optional[State] = None,
        rng: Optional[RNG] = None,
        return_cached: bool = True,
        **data,
    ) -> State:
        ctx = cls(state=state, rng=rng, return_cached=return_cached)
        model(ctx, **data)
        return ctx.result

    def __init__(
        self,
        state: Optional[State] = None,
        rng: Optional[RNG] = None,
        return_cached: bool = True,
    ):
        self.state = state or {}
        self.rng = rng or default_rng()
        self.return_cached = return_cached
        self.result = {}

    def rv(
        self, name: str, dist: BasicDistribution, obs: Optional[Numeric] = None
    ) -> Numeric:
        match self.state.get(name), obs:
            case None, None:
                self.result[name] = dist.sample(rng=self.rng)
                return self.result[name]
            case _, None:
                self.result[name] = self.state[name]
                return self.result[name]
            case None, _:
                return obs
            case _:
                raise RuntimeError("state and obs cannot both be defined.")

    def cached(self, name: str, value: Numeric) -> Numeric:
        """Handle cached values.

        Returns the value `value` and additionally stores `value` in
        `self.result[name]` if the `return_cached` attribute is True.

        Parameters
        ----------
        name : str
            Name of value to cache.
        value : Numeric
            Value of the thing to cache.

        Returns
        -------
        Numeric
            `value`, which is the second argument in `cached`.
        """
        if self.return_cached:
            self.result[name] = value
        return value


class TransformedLogprobAndTrace(Context):
    """TODO.

    Calculates the transformed log probability for a given state (on the real
    space) and also returns the state in the native space.

    Returns
    -------
    tuple[float, State]
        (logprob_plus_logdetjac, native_state_with_cached_items)
    """

    @classmethod
    def run(cls, model, state: State, **data) -> tuple[float, State]:
        ctx = cls(state)

        try:
            model(ctx, **data)
        except NegativeInfinityError:
            # In case of -inf, just return -inf and an empty trace ({}) early.
            # The trace doesn't matter, just need to return something.
            return -np.inf, {}

        return ctx.result

    def __init__(self, state: State):
        self.state = state
        self.result = [0.0, {}]  # logprob, native_state

    def rv(
        self,
        name: str,
        dist: TransformableDistribution,
        obs: Optional[Numeric] = None,
    ):
        if obs is None:
            real_value = self.state[name]
            self.result[0] += np.sum(dist.logpdf_plus_logdetjac(real_value))
            value = dist.to_native(real_value)
            self.result[1][name] = value
        else:
            value = obs
            self.result[0] += np.sum(dist.logpdf(value))

        if self.result[0] == -np.inf:
            raise NegativeInfinityError(
                "Negative infinity in TransformedLogprob."
            )

        return value

    def cached(self, name: str, value: Numeric) -> Numeric:
        self.result[1][name] = value
        return value


class TransformedPredictive(Context):
    """Get transformed predictive state.

    Get transformed predictive state (i.e. state predictive in the real
    space) via the `run` method.

    Parameters
    ----------
    state: Optional[State]
        Contains values on the native space. If a model parameter's
        value is not provided, it will be sampled from it's prior.
        Defaults to None.
    rng: Optional[RNG]
        Random number generator. Defaults to None.
    return_cached: bool
        Whether or not to return cached values. Defaults to True.

    Attributes
    ----------
    state: State
        Contains values on the native space. If None was provided in the
        constructor, it's value will be an empty dictionary.
    rng: RNG
        Random number generator. If None was provided in the
        constructor, this will be `default_rng()`.
    return_cached: bool
        Whether or not to return cached values.
    """

    @classmethod
    def run(
        cls,
        model,
        state: Optional[State] = None,
        rng: Optional[RNG] = None,
        return_cached: bool = True,
        **data,
    ):
        ctx = cls(state=state, rng=rng, return_cached=return_cached)
        model(ctx, **data)
        return ctx.result

    def __init__(
        self,
        state: Optional[State] = None,
        rng: Optional[RNG] = None,
        return_cached: bool = True,
    ):
        self.state = state or {}
        self.rng = rng or default_rng()
        self.return_cached = return_cached
        self.result = {}

    def rv(
        self,
        name: str,
        dist: TransformableDistribution,
        obs: Optional[Numeric] = None,
    ) -> Numeric:
        match self.state.get(name), obs:
            case None, None:
                # Sample from prior.
                native_value = dist.sample(rng=self.rng)
                self.result[name] = dist.to_real(native_value)
                return native_value
            case _, None:
                # provided state is in native space, so needs to be converted
                # to real space.
                self.result[name] = dist.to_real(self.state[name])
                return self.result[name]
            case None, _:
                # Observed values need no transformation.
                return obs
            case _:
                raise RuntimeError("state and obs cannot both be defined.")

    def cached(self, name: str, value: Numeric) -> Numeric:
        """Handle cached values.

        Returns the value `value` and additionally stores `value` in
        `self.result[name]` if the `return_cached` attribute is True.

        Parameters
        ----------
        name : str
            Name of value to cache.
        value : Numeric
            Value of the thing to cache.

        Returns
        -------
        Numeric
            `value`, which is the second argument in `cached`.
        """
        if self.return_cached:
            self.result[name] = value
        return value


class LogprobAndLogjacobianAndTrace(Context):
    @classmethod
    def run(cls, model, state: State, **data) -> tuple[float, float, State]:
        ctx = cls(state)

        try:
            model(ctx, **data)
        except NegativeInfinityError:
            # In case of -inf, just return -inf and an empty trace ({}) early.
            # The trace doesn't matter, just need to return something.
            return -np.inf, -np.inf, {}

        return ctx.result

    def __init__(self, state: State):
        self.state = state
        self.result = [0.0, 0.0, {}]  # logprob, logdetjac, native_state

    def rv(
        self,
        name: str,
        dist: TransformableDistribution,
        obs: Optional[Numeric] = None,
    ):
        if obs is None:
            real_value = self.state[name]
            value = dist.to_native(real_value)
            self.result[0] += np.sum(dist.logpdf(value))
            self.result[1] += np.sum(dist.logdetjac(real_value))
            self.result[2][name] = value
        else:
            value = obs
            self.result[0] += np.sum(dist.logpdf(value))

        if self.result[0] == -np.inf:
            raise NegativeInfinityError(
                "Negative infinity in LogprobAndLogjacobianAndTrace."
            )

        if self.result[1] == -np.inf:
            raise NegativeInfinityError(
                "Negative infinity in LogprobAndLogjacobianAndTrace."
            )

        return value

    def cached(self, name: str, value: Numeric) -> Numeric:
        self.result[2][name] = value
        return value
