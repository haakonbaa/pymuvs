"""
An implementation of the special Euclidean group in 3D,SE(3). Designed to work
with sympy for symbolic computations.

References:

[1] O. Egeland and J. T. Gravdahl, Modeling and simulation for automatic
    control, Corr., 2. print. Trondheim: Marine Cybernetics AS, 2003. ISBN
    82-92356-01-0

[2] T. I. Fossen, Handbook of Marine Craft Hydrodynamics and Motion Control,
    2nd Editioon, Wiley 2011. ISBN 9781119575054
"""

import sympy as sp
import numpy as np
from sympy.matrices import MatrixBase
from numpy import ndarray as NDArray
from typing_extensions import Self
from typing import TypeVar
from deprecated import deprecated

from util import jacobian as _jacobian


_SIMPLIFY: bool = True


def set_simplify(simplify: bool):
    """
    Set global flag to disable/enable simplification of SE3 expressions.
    Having it set to True will simplify the expressions, but may slow down
    computation.
    """
    global _SIMPLIFY
    _SIMPLIFY = simplify


def _simplify(expression: MatrixBase) -> MatrixBase:
    return sp.trigsimp(expression)


class SE3():
    """
    Represents an element of SE(3) - the special Euclidean group in 3D.
    https://en.wikipedia.org/wiki/Euclidean_group
    """

    def __init__(self) -> None:
        self._rotation: MatrixBase = sp.eye(3)
        self._translation: MatrixBase = sp.zeros(3, 1)

    def __repr__(self) -> str:
        return f"SE3(rotation={self._rotation}, translation={self._translation})"

    def __matmul__(self, other: 'SE3') -> 'SE3':
        """
        Multiply two elements of SE(3) together.

        Corresponds to applying the first transformation (other) and then the
        second transformation (self) to a point/vector.
        """
        if not isinstance(other, SE3):
            raise ValueError("Invalid type for multiplication.")

        res = SE3()
        res._rotation = self._rotation @ other._rotation
        res._translation = self._rotation @ other._translation + self._translation
        if _SIMPLIFY:
            res._rotation = _simplify(res._rotation)
            res._translation = _simplify(res._translation)
        return res

    def apply(self, point: MatrixBase | NDArray) -> MatrixBase:
        """
        Applies the transformation to a point or vector.

            [R, t]
        For [0, 1] in SE(3), applies the transformation
            R @ p + t
        to a point p:
        """
        assert isinstance(point, sp.Matrix) or isinstance(point, np.ndarray)
        assert point.shape[0] == 3
        point = sp.Matrix(point)
        expression: MatrixBase = self._rotation @ point + self._translation
        if _SIMPLIFY:
            return _simplify(expression)
        return expression

    def copy(self) -> 'SE3':
        """
        Create a deep copy of the SE3 element.
        """
        r: SE3 = SE3()
        r._rotation = self._rotation.copy()
        r._translation = self._translation.copy()
        return r

    def get_rotation(self) -> MatrixBase:
        """
        Returns the rotation matrix of the SE(3) element.
        """
        return self._rotation

    def get_translation(self) -> MatrixBase:
        """
        Returns the translation vector of the SE(3) element.
        """
        return self._translation

    def get_jacobian(self, q: list[sp.Symbol]) -> MatrixBase:
        """
        Suppose the SE(3) elements transforms from frame b to some inertial
        frame n.
            v^i = T @ v^b
        get_jacobian Returns the matrix mapping from the time derivatives of
        the parameters of T to the twist in the body frame.
            v^b = J(q) @ dq
        """
        assert self.free_symbols().issubset(set(q))
        assert len(q) > 0
        J = sp.zeros(6, len(q))
        R = self._rotation
        # TODO: make it possible to disable simplification
        J[:3, :] = _simplify(
            R.T @ _jacobian(self._translation, sp.Matrix(q)))
        J[3:, :] = rotmat_to_angvel_matrix_frameb(self._rotation, q)

        return J

    def free_symbols(self) -> set[sp.Symbol]:
        return self._rotation.free_symbols.union(self._translation.free_symbols)


def rot_x(theta: sp.Symbol) -> SE3:
    """
    Create an element of SE(3) that represents a rotation around the x-axis.
    Resulting rotation will move a point around the x-axis by theta radians
    following the right-hand rule.
    """
    c = sp.cos(theta)
    s = sp.sin(theta)
    r = SE3()
    r._rotation = sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    return r


def rot_y(theta: sp.Symbol) -> SE3:
    """
    Create an element of SE(3) that represents a rotation around the y-axis.
    Resulting rotation will move a point around the y-axis by theta radians
    following the right-hand rule.
    """
    c = sp.cos(theta)
    s = sp.sin(theta)
    r = SE3()
    r._rotation = sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return r


def rot_z(theta: sp.Symbol) -> SE3:
    """
    Create an element of SE(3) that represents a rotation around the z-axis.
    Resulting rotation will move a point around the z-axis by theta radians
    following the right-hand rule.
    """
    c = sp.cos(theta)
    s = sp.sin(theta)
    r = SE3()
    r._rotation = sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return r


def trans(x: sp.Symbol, y: sp.Symbol, z: sp.Symbol) -> SE3:
    """
    Create an element of SE(3) that represents a translation.
    Resulting translation will move a point by x, y, z units in the x, y, z
    directions respectively.
    """

    t = SE3()
    t._translation = sp.Matrix([x, y, z])
    return t


def inv(T: SE3) -> SE3:
    """
    Invert an element of SE(3).
    Returns a new element of SE(3) that is the inverse of the input.
    T @ inv(T) = 0 (Applying 0 does not change the point)
    """

    r = T.copy()
    r._rotation = r._rotation.T
    r._translation = -r._rotation @ r._translation
    return r


def skew(v: MatrixBase) -> MatrixBase:
    """
    Given a 3x1 vector v, return the 3x3 skew-symmetric matrix S such that
        w x v = S @ v for any 3x1 vector w.
    """
    assert v.shape == (3, 1)
    return sp.Matrix([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])


def down_skew(S: MatrixBase) -> MatrixBase:
    """
    Given a 3x3 skew-symetric matrix S, return the 3x1 vector such that
    w x v = S @ v for any 3x1 vector v.

    Assumes S is skew-symmetric, this is not checked!
    """
    assert S.shape == (3, 3)
    return sp.Matrix([S[2, 1], S[0, 2], S[1, 0]])


def rotmat_to_angvel_matrix_frameb(R: MatrixBase, params: list[sp.Symbol]) -> MatrixBase:
    """
    Given a 3x3 rotation matrix R_b^n(q), compute the matrix J(q) such that
        w_{nb}^b = J(q) * dq

    w_{nb}^b is the angular velocity vector of frame b relative to frame n
    expressed in frame b.

    See equation (6.255) in [1].

    @param R: Rn^b(q) - the rotation matrix from frame n to frame b
    @param params: A list of symbols representing the parameters of R. Note that
        all parameters in R must be present in this list (but not necessarily
        vice versa)

    @return The matrix J(q)
    """

    assert R.shape == (3, 3)
    assert R.free_symbols.issubset(set(params))

    q = sp.Matrix(params)
    dq = sp.Matrix(sp.symbols(f'dq0:{len(params)}'))

    Rdot = sp.Matrix.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            Rdot[i, j] = R[i, j].diff(q).T @ dq

    w = down_skew(R.T @ Rdot)

    # TODO: Assuming the w vector is a linear function of dq. Verify that this
    # is correct. Explot this to create a matrix representation in the standard
    # basis.

    J = sp.zeros(3, len(params))
    for c in range(len(params)):
        subs = {dq[i]: 1 if i == c else 0 for i in range(len(params))}
        J[:, c] = w.subs(subs)

    if _SIMPLIFY:
        J = _simplify(J)

    return J
