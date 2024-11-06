import numpy as np
import sympy as sp
from numpy.typing import NDArray
from util import is_spd

from se3 import SE3

class Position():
    def __init__(self, x: np.float64, y: np.float64, z: np.float64):
        self.x = np.float64(x)
        self.y = np.float64(y)
        self.z = np.float64(z)

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.z])

    def __repr__(self):
        return f"Position(x={self.x}, y={self.y}, z={self.z})"


class Link():
    def __init__(self,
                 mass: np.float64,
                 volume: np.float64,
                 inertia: NDArray[np.float64],
                 added_mass: NDArray[np.float64],
                 linear_damping: NDArray[np.float64],
                 quadratic_damping: NDArray[np.float64],
                 ):
        assert type(inertia) == np.ndarray
        assert inertia.shape == (3, 3)
        assert is_spd(inertia)

        assert type(added_mass) == np.ndarray
        assert added_mass.shape == (6, 6)
        assert is_spd(added_mass)

        assert type(linear_damping) == np.ndarray
        assert linear_damping.shape == (6, 6)
        assert is_spd(linear_damping)

        assert type(quadratic_damping) == np.ndarray
        assert quadratic_damping.shape == (6, 6)
        assert is_spd(quadratic_damping)

        self.mass = np.float64(mass)
        self.volume = np.float64(volume)
        self.inertia = inertia
        self.added_mass = added_mass
        self.linear_damping = linear_damping
        self.quadratic_damping = quadratic_damping

    def __repr__(self) -> str:
        return f"Link(mass={self.mass},\n" \
                    f"volume={self.volume},\n" \
                    f"inertia={self.inertia})\n" \
                    f"added_mass={self.added_mass})\n" \
                    f"linear_damping={self.linear_damping})\n" \
                    f"quadratic_damping={self.quadratic_damping})"


class Robot():
    def __init__(self, links: list[Link], transforms: list[SE3]):
        self.links = links
        self.transforms = transforms
