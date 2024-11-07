import sympy as sp
import numpy as np
from sympy.matrices import MatrixBase
from numpy import ndarray as NDArray
from typing_extensions import Self
from typing import TypeVar

_SIMPLIFY: bool = True


def set_simplify(simplify: bool):
    global _SIMPLIFY
    _SIMPLIFY = simplify


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
            res._rotation = sp.simplify(res._rotation)
            res._translation = sp.simplify(res._translation)
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
            return sp.simplify(expression)
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

def rotmat_to_angular_velocity(R: MatrixBase,
                               params: list[sp.Symbol],
                               diff_params: list[sp.Symbol]) -> MatrixBase:
    """
    Given a rotation matrix R, compute the angular velocity vector w as a function
    of the parameters and their time derivatives.

    @param R: The rotation 3x3 matrix
    @param params: A list of symbols representing the parameters. Note that all
        parameters in R must be present in this list (but not necessarily vice versa)
    @param diff_params: A list of symbols representing the time derivatives of
        the parameters in the same order as in params.

    Note: R is assumed to be a 3x3 rotation matrix, this is not checked!
    """

    # TODO: Fix!

    assert R.shape == (3, 3)
    assert len(params) == len(diff_params)
    assert R.free_symbols.issubset(set(params))

    q = sp.Matrix(params)
    dq = sp.Matrix(diff_params)

    R_diff = sp.Matrix.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            #print(R[i, j].diff(q).T @ dq)
            #exit()
            R_diff[i, j] = R[i, j].diff(q).T @ dq

    S = R_diff @ R.T
    # TODO: make sure S is skew-symmetric
    w = sp.Matrix([S[2, 1], S[0, 2], S[1, 0]])
    if _SIMPLIFY:
        w = sp.simplify(w)
    return w

def down_skew(S: MatrixBase) -> MatrixBase:
    """
    Given a 3x3 skew-symetric matrix S, return the 3x1 vector such that
    w x v = S @ v for any 3x1 vector v.

    Assumes S is skew-symmetric, this is not checked!
    """
    assert S.shape == (3, 3)
    return sp.Matrix([S[2, 1], S[0, 2], S[1, 0]])

def rotmat_to_angvel_matrix(R: MatrixBase, params: list[sp.Symbol]) -> MatrixBase:

    assert R.shape == (3, 3)
    assert R.free_symbols.issubset(set(params))

    R = R.T # REMOVE

    q = sp.Matrix(params)
    dq = sp.Matrix(sp.symbols(f'dq0:{len(params)}'))

    Rdot = sp.Matrix.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            Rdot[i, j] = R[i, j].diff(q).T @ dq
    w = down_skew(Rdot @ R.T)
    if _SIMPLIFY:
        w = sp.simplify(w)

    J = sp.zeros(3, len(params))
    for c in range(len(params)):
        subs = {dq[i]: 1 if i == c else 0 for i in range(len(params))}
        J[:, c] = w.subs(subs)

    return -J  # J

def rotmat_to_angvel_matrix_v2(R: MatrixBase, params: list[sp.Symbol]) -> MatrixBase:

    assert R.shape == (3, 3)
    assert R.free_symbols.issubset(set(params))

    q = sp.Matrix(params)
    dq = sp.Matrix(sp.symbols(f'dq0:{len(params)}'))

    Rdot = sp.Matrix.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            Rdot[i, j] = R[i, j].diff(q).T @ dq
    w = down_skew(Rdot @ R.T)
    if _SIMPLIFY:
        w = sp.simplify(w)

    J = sp.zeros(3, len(params))
    for c in range(len(params)):
        subs = {dq[i]: 1 if i == c else 0 for i in range(len(params))}
        J[:, c] = w.subs(subs)

    return J 
   

if __name__ == "__main__":
    # run some tests
    # Create a new element of SE(3)


    r, p, y = sp.symbols('ϕ θ ψ')
    dr, dp, dy = sp.symbols('dϕ dθ dψ')
    T = rot_z(y) @ rot_y(p) @ rot_x(r)

    # simple rotations
    Rx = rot_x(r)
    Sx = rotmat_to_angvel_matrix(Rx.get_rotation(), [r, p, y])
    print(Sx)
    print(Sx @ sp.Matrix([dr, dp, dy]))

    Ry = rot_y(p)
    Sy = rotmat_to_angvel_matrix(Ry.get_rotation(), [r, p, y])
    print(Sy)
    print(Sy @ sp.Matrix([dr, dp, dy]))

    Rz = rot_z(y)
    Sz = rotmat_to_angvel_matrix(Rz.get_rotation(), [r, p, y])
    print(Sz)
    print(Sz @ sp.Matrix([dr, dp, dy]))
    #T = rot_x(r) @ rot_y(p) @ rot_z(y)
    #T = rot_x(r)
    #T = rot_y(p)
    #T = rot_z(y)

    R = T.get_rotation()
    q = sp.Matrix([r, p, y])

    S = rotmat_to_angvel_matrix_v2(R, [r, p, y])
    w = S @ sp.Matrix([dr, dp, dy])
    c1 = {r: 0, p: sp.pi/2, y: 0, dr: 1, dp: 0, dy: 0}

    print(S)
    print(w)

    print(S.subs(c1))
    print(w.subs(c1))

    #w = rotmat_to_angular_velocity(R, [r, p, y], [dr, dp, dy])














