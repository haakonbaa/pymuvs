import numpy as np
import sympy as sp
from sympy.matrices import MatrixBase
from util import is_spd, is_symmetric
from numpy.typing import NDArray

from se3 import SE3, rotmat_to_angvel_matrix_frameb

class Model():
    """
    A mathematical model of a robot on the form
        M(q) * ddq + C(q, dq) dq + D(q, dq) dq + g(q) = tau
    """
    #TODO: make sure it is C(q, dq) dq and small g(q) everywhere in the code.
    def __init__(self, M: MatrixBase, C: MatrixBase, D: MatrixBase, g: MatrixBase):
        assert M.shape[0] == M.shape[1]
        n = M.shape[0]
        assert C.shape == (n, n)
        assert D.shape == (n, n)
        assert g.shape == (n, 1)
        self.M = M
        self.C = C
        self.D = D
        self.g = g


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
        assert is_symmetric(inertia)

        # TODO: Perform some checks on the added mass linear and quadratic
        # damping
        assert type(added_mass) == np.ndarray
        assert added_mass.shape == (6, 6)
        assert is_symmetric(added_mass)

        assert type(linear_damping) == np.ndarray
        assert linear_damping.shape == (6, 6)
        assert is_symmetric(linear_damping)

        assert type(quadratic_damping) == np.ndarray
        assert quadratic_damping.shape == (6, 6)
        assert is_symmetric(quadratic_damping)

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
                raise ValueError("Extra parameters in the params list: "
                                 f"{param_symbols - transform_symbols}. "
                                 "These parameters were not found in the transforms.")
            else:
                # TODO: Might want to remove this check.
                # People might want to have parameterized constants in the
                # transforms.
                raise ValueError("Missing paramaters in the params list: "
                                 f"{transform_symbols - param_symbols}. "
                                 "These parameters were not found in the transforms.")

        for diff_param in diff_params:
            if diff_param in param_symbols:
                raise ValueError(f"Diff param {diff_param} found in params "
                                 "list. These symbols should be unique!")

        self._links = links
        self._transforms = transforms
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

    def get_link_count(self) -> int:
        """
        Returns the number of links in the robot.
        """
        return len(self._links)

    def get_jacobian(self) -> MatrixBase:
        """
        Returns the Jacobian matrix of the robot.

        The Jacobian matrix, J(q), maps the generalized velocities dq to the linear
        and angular velocities of each link in the robot.
            v = J(q) * dq
        J is a (6 * link_count) x (DOF) matrix
        """

        nl = self.get_link_count()
        ndof = self.get_dof()

        # Jacobian matrix
        J = sp.zeros(6 * nl, ndof)
        for link_num in range(nl):
            Ji = sp.zeros(6, ndof)

            p = self._transforms[link_num].get_translation()
            R = self._transforms[link_num].get_rotation()

            Ji[:3, :] = _jacobian(p, self._q)
            Ji[3:, :] = rotmat_to_angvel_matrix_frameb(R, self._params)

            J[6*link_num:6*(link_num+1), :] = Ji

        return J

    def get_model(self, gvec : NDArray[np.float64] = np.array([0,0,-9.81])) -> None:
        """
        Returns the robot model as a Model object. The model object represents
        a mathematical model on the form
            M(q) * ddq + C(q, dq) dq + D(q, dq) + G(q) = tau
        """

        # Will formulate the enrgy as
        # K = 0.5 * dq^T J(q)^T M J(q) dq

        # The potential enegy is compensated for by modelling graviy and buoyancy
        # as forces acting on the links instead of potential energy.
        # TODO: model forces acting on the links

        if isinstance(gvec, np.ndarray):
            assert gvec.size == 3
            gvec.reshape((3, 1))

        J = self.get_jacobian()

        # Mass matrix
        M = sp.zeros(6*self.get_link_count(), 6*self.get_link_count())
        for link_num in range(self.get_link_count()):
            link = self._links[link_num]
            mass = link.mass
            inertia = link.inertia
            added_mass = link.added_mass

            # TODO: adjust inertia and added mass based on center of mass.
            # for now assume the center of mass is at the origin.

            mi = 6*link_num
            M[mi:mi+3, mi:mi+3] = sp.eye(3) * mass
            M[mi+3:mi+6, mi+3:mi+6] = inertia
            M[mi:mi+6, mi:mi+6] += added_mass

        Ma = sp.simplify(J.T * M * J)

        # time derivative of Jacobian
        Jd = _time_diff_matrix(J, self._params, self._diff_params)

        # Coreolis matrix
        C = sp.zeros(self.get_dof(), self.get_dof())

        C += J.T * M * Jd + Jd.T * M * J # TODO: verify this is correct

        ugly = J.T * M * J * self._dq
        C += 0.5 * _jacobian(ugly, self._q).T
        C = sp.simplify(C)

        # gravity and buoyancy

        g = sp.zeros(self.get_dof(), 1)
        for link_num, link in enumerate(self._links):
            Tbn = self._transforms[link_num]
            pos = Tbn.get_translation()
            for i in range(self.get_dof()):
                # TODO: adjust for center of mass
                qi = sp.Matrix([self._params[i]])
                _jacobian(pos, qi)
                g[i] += _jacobian(pos, sp.Matrix([self._params[i]])).T @ gvec * link.mass

        g = - sp.simplify(g)

        """
        print(f"{g=}")
        print(f"{M=}")
        print(f"{J=}")
        print(f"{Jd=}")
        print(f"{Ma=}")
        print(f"{C=}")
        """

        return Model(Ma, C, sp.zeros(self.get_dof(), self.get_dof()), g)



def _time_diff_matrix(A : MatrixBase, q: list[sp.Symbol], dq: list[sp.Symbol]) -> MatrixBase:
    """
    Compute the time derivative of a matrix A. A is a function of q, and the
    time derivatives of q are given by dq.
    """
    assert len(q) == len(dq)

    Ad = sp.zeros(A.shape[0], A.shape[1])
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            Ad[r, c] = A[r, c].diff(sp.Matrix(q)).T @ sp.Matrix(dq)

    return Ad


def _jacobian(x: MatrixBase, q: MatrixBase) -> MatrixBase:
    """
    Compute the Jacobian of a nx1 vector x with respect to a list of
    parameters q.
    """

    assert ((x.shape[0] == 1) or (x.shape[1] == 1))
    assert q.shape[1] == 1

    if (x.shape[0] == 1) and (x.shape[1] != 1):
        return _jacobian(x.T, q).T

    nx = x.shape[0]
    nq = q.shape[0]

    J = sp.zeros(nx, nq)
    for i in range(nx):
        J[i, :] = x[i].diff(q).T

    return J


def main():
    x, y, z = sp.symbols('x y z')
    m = sp.Matrix([[x+y**2 + x*z], [x*y*z]])
    q = sp.Matrix([x, y, z])
    J = _jacobian(m, q)
    print(J, J.shape)


if __name__ == "__main__":
    main()
