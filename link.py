import numpy as np
import sympy as sp
from sympy.matrices import MatrixBase
from util import is_spd
from numpy.typing import NDArray

from se3 import SE3


"""
class Position():
    def __init__(self, x: np.float64, y: np.float64, z: np.float64):
        self.x = np.float64(x)
        self.y = np.float64(y)
        self.z = np.float64(z)

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.z])

    def __repr__(self):
        return f"Position(x={self.x}, y={self.y}, z={self.z})"
"""


class Link():
    """
    Represents a rigid body link in a robot.
    """

    def __init__(self,
                 mass: np.float64 | float,
                 volume: np.float64 | float,
                 inertia: NDArray[np.float64],
                 added_mass: NDArray[np.float64],
                 linear_damping: NDArray[np.float64],
                 quadratic_damping: NDArray[np.float64],
                 center_of_mass: NDArray[np.float64] = np.zeros((3,)),
                 center_of_buoyancy: NDArray[np.float64] = np.zeros((3,))
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
        self.center_of_mass = center_of_mass
        self.center_of_buoyancy = center_of_buoyancy

    def __repr__(self) -> str:
        return f"Link(mass={self.mass},\n" \
            f"volume={self.volume},\n" \
            f"inertia={self.inertia})\n" \
            f"added_mass={self.added_mass})\n" \
            f"linear_damping={self.linear_damping})\n" \
            f"quadratic_damping={self.quadratic_damping})"


class Robot():
    def __init__(self, links: list[Link],
                 transforms: list[SE3],
                 params: list[sp.Symbol],
                 diff_params: list[sp.Symbol]):
        """
        A Robot is a collection of links and transformations describing their
        relative positions and orientations.

        @param links: A list of Link objects.
        @param transforms: A list of SE3 objects.
        @param variables: A list of sympy symbols representing the parameters.
            This list should contain exactly the same symbols as in all transforms.
        @param diff_variables: A list of sympy symbols representing the differential parameters.
            This list should contain the derivative of each symbol in @param variables.
            and in the same order.

        We denote by q the generalized coordinates (defined by the params list)
        and dq the time derivative of q (q_dot or \dot{q}) if you will).
        """
        assert len(links) == len(transforms)
        assert len(params) == len(diff_params)
        for var in params:
            assert isinstance(var, sp.Symbol)
        for diff_var in diff_params:
            assert isinstance(diff_var, sp.Symbol)

        # Make sure all parameters used in the transforms are present in the
        # params list.
        transform_symbols = set()
        param_symbols = set(params)
        for t in transforms:
            transform_symbols.update(t.free_symbols())

        if transform_symbols != param_symbols:
            if transform_symbols.issubset(param_symbols):
                raise ValueError("Extra parameters in the params list: " \
                        f"{param_symbols - transform_symbols}. " \
                        "These parameters were not found in the transforms.")
            else:
                raise ValueError("Missing paramaters in the params list: " \
                        f"{transform_symbols - param_symbols}. " \
                        "These parameters were not found in the transforms.")

        for diff_param in diff_params:
            if diff_param in param_symbols:
                raise ValueError(f"Diff param {diff_param} found in params " \
                                 "list. These symbols should be unique!")

        self._links = links
        self._transforms = transforms
        self._link_count: int = len(links)
        self._params = params
        self._diff_params = diff_params
        self._q = sp.Matrix(params)
        self._dq = sp.Matrix(diff_params)

    def get_dof(self) -> int:
        """
        Returns the number of degrees of freedom of the robot. The number of
        generalized coordinates typically denoted by q.
        """
        return len(self._params)

    def get_jacobian(self) -> MatrixBase:
        """
        Returns the Jacobian matrix of the robot.

        The Jacobian matrix, J(q), maps the generalized velocities dq to the linear
        and angular velocities of each link in the robot.
            v = J(q) * dq
        J is a (6 * link_count) x dof matrix
        """

    def get_link_count(self) -> int:
        return self._link_count

    def get_model(slef):
        pass
