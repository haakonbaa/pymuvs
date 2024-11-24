import numpy as np
import sympy as sp
from sympy.matrices import MatrixBase
from numpy.typing import NDArray

from .se3 import SE3, rotmat_to_angvel_matrix_frameb, trans, set_simplify
from .util import jacobian as _jacobian, is_symmetric, is_spd, skew


class Model():
    """
    A mathematical model of a robot on the form
        M(q) * ddq + C(q, dq) dq + D(q, dq) dq + g(q) = Jf(q) B u(z)
    z is a specified vector of inputs to the robot.
    """
    # TODO: make sure it is C(q, dq) dq and small g(q) everywhere in the code.

    def __init__(self,
                 M: MatrixBase,
                 C: MatrixBase,
                 D: MatrixBase,
                 g: MatrixBase,
                 J: MatrixBase,
                 Jf: MatrixBase,
                 B: MatrixBase,
                 u: MatrixBase,
                 transforms: list[SE3],
                 params: list[sp.Symbol],
                 diff_params: list[sp.Symbol],
                 inputs: list[sp.Symbol]):
        assert M.shape[0] == M.shape[1]
        n = M.shape[0]
        assert C.shape == (n, n)
        assert D.shape == (n, n)
        assert g.shape == (n, 1)
        assert J.shape[1] == n
        assert Jf.shape[0] == n
        assert Jf.shape[1] == B.shape[0]
        assert B.shape[1] == u.shape[0]
        self.M = M
        self.C = C
        self.D = D
        self.g = g
        self.J = J
        self.Jf = Jf
        self.B = B
        self.u = u
        self.transforms = transforms
        self.params = params
        self.diff_params = diff_params
        self.inputs = inputs

    def eval(self, q, dq, z=None):
        """
        returns ddq = M^-1 * (tau - C(q, dq) dq - D(q, dq) dq - g(q))
        """
        assert len(q) == len(self.params)
        assert len(dq) == len(self.diff_params)
        if z is None:
            z = np.zeros((len(self.inputs),))
        assert len(z) == len(self.inputs)
        assert isinstance(q, np.ndarray)
        assert isinstance(dq, np.ndarray)

        q = q.reshape((-1, 1))
        dq = dq.reshape((-1, 1))
        z = z.reshape((-1, 1))

        subs_q = {self.params[i]: float(q[i, 0]) for i in range(len(q))}
        subs_dq = {self.diff_params[i]: float(
            dq[i, 0]) for i in range(len(dq))}
        subs_z = {self.inputs[i]: float(z[i, 0]) for i in range(len(z))}

        M = np.array(self.M.subs(subs_q), dtype=np.float64)
        M_inv = np.linalg.inv(M)
        C = np.array(self.C.subs(subs_q).subs(subs_dq), dtype=np.float64)
        D = np.array(self.D.subs(subs_q).subs(subs_dq), dtype=np.float64)
        g = np.array(self.g.subs(subs_q), dtype=np.float64)
        Jf = np.array(self.Jf.subs(subs_q), dtype=np.float64)
        B = np.array(self.B, dtype=np.float64)
        u = np.array(self.u.subs(subs_z), dtype=np.float64)

        return (M_inv @ (Jf @ B @ u - C @ dq - D @ dq - g)).flatten()


class Wrench():
    """
    A wrench is a force and a torque acting on a point in space.
    """

    def __init__(self, position: NDArray[np.float64],
                 wrench: sp.Matrix):
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)

        assert isinstance(wrench, sp.Matrix)  # TODO: support numpy arrays?
        assert wrench.shape == (6, 1)

        self.position = position
        self.wrench = wrench

    def get_force_and_torque(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Returns the force and torque of the wrench.
        """
        return self.wrench[0:3, 0], self.wrench[3:, 0]

    def __repr__(self) -> str:
        return f"Wrench(position={self.position}, wrench={self.wrench})"


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
                 center_of_buoyancy: NDArray[np.float64] = np.zeros((3,)),
                 wrenches: list[Wrench] = [],
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
        self.wrenches = wrenches

    def __repr__(self) -> str:
        return f"Link(mass={self.mass},\n" \
            f"volume={self.volume},\n" \
            f"inertia={self.inertia})\n" \
            f"added_mass={self.added_mass})\n" \
            f"linear_damping={self.linear_damping})\n" \
            f"quadratic_damping={self.quadratic_damping})"

    def get_mass_matrix(self) -> NDArray[np.float64]:
        """
        Returns the mass matrix of the link.
        """
        M = np.zeros((6, 6))
        M[0:3, 0:3] = np.eye(3) * self.mass
        M[3:6, 3:6] = self.inertia
        M += self.added_mass
        # Using Fossen eq 3.27 page 60 we shift the spatial inertia matrix
        # to be about the link's coordinate system, not center of mass.
        # TODO: Verify that this is correct. (run some tests)
        # TODO: cite this in the README
        v = self.center_of_mass
        S = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        I = np.eye(3)
        H = np.block([[I, S.T],
                      [np.zeros((3, 3)), I]])

        return H.T @ M @ H


class Robot():
    def __init__(self, links: list[Link],
                 transforms: list[SE3],
                 params: list[sp.Symbol],
                 diff_params: list[sp.Symbol],
                 inputs: list[sp.Symbol] = [],):
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
        @param inputs: A list of sympy symbols representing the inputs to the robot.
            These are typically the control inputs (like forces and torques).

        We denote by q the generalized coordinates (defined by the params list)
        and dq the time derivative of q (q_dot or \dot{q}) if you will).
        """
        # TODO: make these checks more complete / informative.
        assert len(links) == len(transforms)
        assert len(params) == len(diff_params)
        for var in params:
            assert isinstance(var, sp.Symbol)
        for diff_var in diff_params:
            assert isinstance(diff_var, sp.Symbol)
        for inp in inputs:
            assert isinstance(inp, sp.Symbol)
        for link in links:
            assert isinstance(link, Link)
            for wrench in link.wrenches:
                assert wrench.wrench.free_symbols.issubset(set(inputs))

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
        self._inputs = inputs

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

    def get_model(self, gvec: NDArray[np.float64] = np.array([0, 0, -9.81]),
                  bvec: NDArray[np.float64] = np.array([0, 0, 0]),
                  simplify: bool = True
                  ) -> None:
        """
        Returns the robot model as a Model object. The model object represents
        a mathematical model on the form
            M(q) ddq + C(q, dq) dq + D(q, dq) dq + g(q) = Jf(q) B u(z)

        @param gvec: The gravity vector acting on the robot. Normal value is
            [0, 0, -9.81] m/s^2. This vector is multiplied by the mass to get
            the gravitational force acting on each link.
        @param bvec: The buoyancy vector acting on the robot. This vector is
            multiplied by the volume of each link to get the buoyancy force
            acting on each link. Normal value is
            [0, 0, 1025 kg/m^3 * 9.81 m/s^2] N/m^3 when volume is given in m^3.
            [0, 0, 1.025 kg/L * 9.81 m/s^2] N/m^3 when volume is given in L.
        """

        # Will formulate the enrgy as
        # K = 0.5 * dq^T J(q)^T M J(q) dq

        # The potential enegy is compensated for by modelling graviy and buoyancy
        # as forces acting on the links instead of potential energy.
        # TODO:
        # - [ ] add possibility for external forces and torques

        if isinstance(gvec, np.ndarray):
            assert gvec.size == 3
            gvec.reshape((3, 1))

        # stack the jacobian matrices
        print('jacobian')
        J = sp.zeros(6 * self.get_link_count(), self.get_dof())
        for t_num, T in enumerate(self._transforms):
            print(f'Jacobian for link {t_num}:')
            link = self._links[t_num]
            mi = 6 * t_num
            J[mi:mi+6, :] = T.get_jacobian(self._params)

        # Mass and damping matrix
        print('mass matrix')
        M = sp.zeros(6*self.get_link_count(), 6*self.get_link_count())
        Dlin = sp.zeros(6*self.get_link_count(), 6*self.get_link_count())
        Dquad = sp.zeros(6*self.get_link_count(), 6*self.get_link_count())

        for link_num in range(self.get_link_count()):
            link = self._links[link_num]
            mi = 6*link_num
            M[mi:mi+6, mi:mi+6] = link.get_mass_matrix()
            #mass = link.mass
            #inertia = link.inertia
            #added_mass = link.added_mass

            #M[mi:mi+3, mi:mi+3] = sp.eye(3) * mass
            #M[mi+3:mi+6, mi+3:mi+6] = inertia
            #M[mi:mi+6, mi:mi+6] += added_mass

            Dlin[mi:mi+6, mi:mi+6] = link.linear_damping
            Dquad[mi:mi+6, mi:mi+6] = link.quadratic_damping

            # TODO: Verify that the calculation of the "quadratic" damping is
            # correct.
            Dquad_i = link.quadratic_damping
            abs_twist = sp.Abs(J[mi:mi+6, :] @ self._dq)
            if simplify:
                abs_twist = sp.simplify(abs_twist)
            for i in range(6):
                Dquad_i[i, :] = Dquad_i[i, :] * abs_twist[i]
            Dquad[mi:mi+6, mi:mi+6] = Dquad_i

        Ma = J.T * M * J
        if simplify:
            Ma = sp.simplify(Ma)
        # TODO: This implies the damping forces are applied to the center of
        # mass, maybe applying them to the center of buoyancy is more correct?
        D = J.T * (Dlin + Dquad) * J
        if simplify:
            D = sp.simplify(D)

        # time derivative of Jacobian
        print('time derivative of Jacobian')
        Jd = _time_diff_matrix(J, self._params, self._diff_params)

        # Coreolis matrix
        print('Coreolis matrix')
        C = sp.zeros(self.get_dof(), self.get_dof())
        C += J.T * M * Jd + Jd.T * M * J  # TODO: verify this is correct
        ugly = J.T * M * J * self._dq
        C -= 0.5 * _jacobian(ugly, self._q).T
        if simplify:
            C = sp.simplify(C)

        # gravity and buoyancy

        print('gravity and buoyancy')
        g = sp.zeros(self.get_dof(), 1)
        for link_num, link in enumerate(self._links):
            Tbn_g = self._transforms[link_num] @ trans(*link.center_of_mass)
            pos_g = Tbn_g.get_translation()

            # buoyancy force
            Tbn_b = self._transforms[link_num] @ trans(
                *link.center_of_buoyancy)
            pos_b = Tbn_b.get_translation()

            for i in range(self.get_dof()):
                # TODO: I think this can be written more elegantly with a
                # jacobian.
                qi = sp.Matrix([self._params[i]])
                g[i] += _jacobian(pos_g, qi).T @ gvec * link.mass
                g[i] += _jacobian(pos_b, qi).T @ bvec * link.volume

        # move to the other side of the equation
        g = - g
        if simplify:
            g = sp.simplify(g)

        # Forces and torques
        # = Jf(q) B u(z)    (z is vector of inputs)
        Bu = sp.zeros(self.get_link_count()*6, 1)
        Jf = J.T
        for link_num, link in enumerate(self._links):
            for wrench in link.wrenches:
                force, torque = wrench.get_force_and_torque()
                torque_co = torque + skew(wrench.position) @ force
                Bu[6*link_num:6*link_num+3, 0] += force
                Bu[6*link_num+3:6*link_num+6, 0] += torque_co

        B, u = _Bu_to_B_and_u(Bu)

        return Model(Ma, C, D, g, J, Jf, B, u,
                     self._transforms, self._params, self._diff_params,
                     self._inputs)


def _Bu_to_B_and_u(M: sp.Matrix):
    # collapse the matrix into a column vector, if it is not already
    K = sp.zeros(M.shape[0], 1)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            K[r] += M[r, c]

    u = []
    index_of_factor = {}
    cflistlist = []
    for r in range(K.shape[0]):
        # A list of pairs of (coefficient, factor)'s. i.e [(4, force_1), ...]
        # for each row in K.
        cflist = []
        addends = K[r].as_ordered_terms()
        for addend in addends:
            coeff, factors = addend.as_coeff_mul()
            if coeff != 0:
                cflist.append((coeff, sp.Mul(*factors)))
        cflistlist.append(cflist)

        # Kepp track of the factors and their indices, this will be used to
        # construct u and the B matrix.
        for coeff, factor in cflist:
            if (factor not in u):
                u.append(factor)
                index_of_factor[factor] = len(u) - 1

    if len(u) == 0:
        return sp.zeros(K.shape[0], 1), sp.zeros(1, 1)

    B = sp.zeros(K.shape[0], len(u))
    for r, cflist in enumerate(cflistlist):
        for coeff, factor in cflist:
            B[r, index_of_factor[factor]] = coeff
    return B, sp.Matrix(u)


def _time_diff_matrix(A: MatrixBase, q: list[sp.Symbol], dq: list[sp.Symbol]) -> MatrixBase:
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
