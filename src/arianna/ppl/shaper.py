import numpy as np

from arianna.types import State


class Shaper:
    """Shapes dict of numeric values into np.array and back."""

    @classmethod
    def from_state(cls, state: State):
        """Construct a Shaper from a state."""
        return cls({name: np.shape(value) for name, value in state.items()})

    def __init__(self, shape: dict[str, tuple[int, ...]]):
        self.shape = shape
        self.dim = int(sum(np.prod(s) for s in self.shape.values()))

    def vec(self, state: State) -> np.ndarray:
        """Convert a state dict into a np.ndarray."""
        flat_state = []
        for _, value in state.items():
            value = np.array(value)
            flat_state.extend(value.flatten())
        return np.array(flat_state)

    def unvec(self, flat_state: np.ndarray) -> State:
        """Convert a np.ndarray back to a state dict."""
        state = {}
        start = 0
        for name, shapes in self.shape.items():
            num_elems = int(np.prod(shapes))
            value = np.reshape(flat_state[start : start + num_elems], shapes)
            state[name] = value
            start += num_elems
        return state
